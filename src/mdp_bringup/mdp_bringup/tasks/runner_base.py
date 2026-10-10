"""What task1_runner and task2_runner share: the topics and services every run
has, the EKF pose in the `map` frame, the run timer and the tablet-cell format.

A runner subclasses RunnerBase and provides:
  control_loop()          20 Hz; the base publishes /run_timer before it and
                          /run_status (publish_run_status()) at 2 Hz
  publish_run_status()    fills and sends mdp_interfaces/RunStatus
  start_run_callback / stop_run_callback   /start_run, /stop_run (std_srvs/Trigger)
  set_pose_callback       /set_pose - the EKF being reset (/reset_pose)
  yolo_callback           /yolo_result
  on_pose()               optional: after each new pose (e.g. reset confirmation)
and creates self.follower (after its parameters are set) before spinning.
"""
import math

import rclpy.time
import tf2_geometry_msgs  # noqa: F401  registers the geometry_msgs conversions tf2 uses
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from mdp_interfaces.msg import RunStatus
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_localization.srv import SetPose
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from mdp_algorithm.control.path_follower import yaw_from_quaternion
from mdp_bringup.utils import markers
from mdp_bringup.utils.run import wall_timer

STATUS_PERIOD_S = 0.5      # /run_status at 2 Hz; /run_timer every loop


class RunnerBase(Node):
    def __init__(self, name: str):
        super().__init__(name)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bt_pub = self.create_publisher(String, '/bluetooth_tx', 10)
        self.run_status_pub = self.create_publisher(RunStatus, '/run_status', 10)
        self.timer_pub = self.create_publisher(Marker, '/run_timer', 10)
        # Published once per plan: TRANSIENT_LOCAL so Foxglove connecting later still gets them.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', latched)
        self.arena_pub = self.create_publisher(MarkerArray, '/grid_markers', latched)
        self.obstacle_pub = self.create_publisher(MarkerArray, '/obstacle_markers', latched)
        self.checkpoint_pub = self.create_publisher(MarkerArray, '/checkpoint_markers', latched)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.leg_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        self.search_pub = self.create_publisher(MarkerArray, '/search_progress', 10)

        self.create_subscription(String, '/yolo_result', self.yolo_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/set_pose', self.set_pose_callback, 10)
        self.create_service(Trigger, '/start_run', self.start_run_callback)
        self.create_service(Trigger, '/stop_run', self.stop_run_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ekf_shift = self.create_client(SetPose, '/set_pose')   # shift_pose(); the service, no RESET
        wall_timer(self, 0.05, self._loop)   # 20 Hz

        self.follower = None             # the runner's PathFollower
        self.current_pose = (0.0, 0.0, 0.0)   # map frame, from odom_callback
        self.have_pose = False
        self.last_odom = None
        self.state = None
        self.state_start = self.now()
        self.run_start = self.run_end = None  # GO and FINISHED/STOP times, for run_time
        self._last_status = 0.0
        self._last_cmd = (0.0, 0.0)

    # ------------------------------------------------------------ helpers ----

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def stamp(self):
        return self.get_clock().now().to_msg()

    def set_state(self, state):
        self.state = state
        self.state_start = self.now()

    def run_time(self) -> float:
        """Seconds since GO; frozen at FINISHED / STOP."""
        if self.run_start is None:
            return 0.0
        return (self.run_end if self.run_end is not None else self.now()) - self.run_start

    def end_run(self):
        """FINISHED / STOP: freeze the run time."""
        if self.run_start is not None and self.run_end is None:
            self.run_end = self.now()

    @staticmethod
    def fmt(pose) -> str:
        """Tablet cell + the nearest of N/E/S/W: '(5,8)E'."""
        return f"({markers.cell(pose[0])},{markers.cell(pose[1])}){markers.direction(pose[2])}"

    def send_cmd(self, v: float, w: float):
        msg = TwistStamped()
        msg.header.stamp = self.stamp()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)
        self._last_cmd = (float(v), float(w))

    def send_bt(self, text: str):
        self.bt_pub.publish(String(data=text))

    # ---------------------------------------------------------------- pose ----

    def odom_callback(self, msg: Odometry):
        """EKF pose (odom frame) -> map frame via TF. On a lookup failure keep the
        previous pose: the raw odom pose would be off by the whole start pose."""
        try:
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"POSE      no map <- {msg.header.frame_id} transform yet ({exc})",
                                   throttle_duration_sec=2.0)
            return
        self.last_odom = msg              # odom frame, for a pose correction (task1 IR fix)
        pose = tf2_geometry_msgs.do_transform_pose(msg.pose.pose, tf)
        self.current_pose = (pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation))
        self.have_pose = True
        if self.follower is not None:
            self.follower.update_pose(*self.current_pose)
        self.on_pose()

    def on_pose(self):
        pass

    def shift_pose(self, dx: float, dy: float) -> bool:
        """Move the EKF pose by (dx, dy) in the map frame, heading kept (a sensor
        fix). False when it cannot (no odom yet / no odom <- map transform)."""
        if self.last_odom is None:
            return False
        try:   # the shift is in the map frame; the EKF wants odom
            tf = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
        except tf2_ros.TransformException:
            return False
        a = yaw_from_quaternion(tf.transform.rotation)
        odom = self.last_odom.pose.pose
        req = SetPose.Request()
        req.pose.header.frame_id = self.last_odom.header.frame_id
        req.pose.header.stamp = self.last_odom.header.stamp   # the EKF's own time base
        p = req.pose.pose.pose
        p.position.x = odom.position.x + math.cos(a) * dx - math.sin(a) * dy
        p.position.y = odom.position.y + math.sin(a) * dx + math.cos(a) * dy
        p.orientation = odom.orientation
        req.pose.pose.covariance = list(self.last_odom.pose.covariance)
        self.ekf_shift.call_async(req)
        return True

    # ---------------------------------------------------------------- loop ----

    def _loop(self):
        self.publish_timer()
        now = self.now()
        if now - self._last_status >= STATUS_PERIOD_S:
            self._last_status = now
            self.publish_run_status()
        self.control_loop()

    def publish_timer(self):
        """/run_timer every loop (20 Hz), so the 3D view counts smoothly."""
        phase = 'idle' if self.run_start is None else 'running' if self.run_end is None else 'done'
        self.timer_pub.publish(markers.run_timer(self.stamp(), self.run_time(), phase))
