"""
Task 2 runner: drive the slalom around the two obstacles and back to the carpark.

  WAITING_FOR_START --/start_run--> DRIVING_PATH --> FINISHED

At `go` it builds the waypoints from the first arrow YOLO saw (38 = right,
39 = left: which side of obstacle 1 to pass), fits one smooth spline through
them (mdp_algorithm spline_planner) and follows it with pure pursuit, slowing
down in tight curves. The blocks come from config/tasks.yaml (task2) - the
same file the Gazebo arena is built from. Pose: /odometry/filtered converted
into `map` via TF.
Drawn in Foxglove: /planned_path, /waypoint_markers (arena boxes, waypoints).
"""

import math
from enum import Enum, auto

import tf2_geometry_msgs  # noqa: F401  registers the geometry_msgs conversions tf2 uses
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from mdp_algorithm.control.pure_pursuit_follower import yaw_from_quaternion
from mdp_algorithm.planning.spline_planner import SplinePathPlanner
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils.run import run, wall_timer
from mdp_bringup.utils import config, markers, obstacle_layout

CAR = planner_params.car_from_urdf()   # wheelbase, steering limits: the URDF
# Per-side curvature limits (1/m): left 43 deg, right 32.5 deg.
KAPPA_LEFT = math.tan(CAR["steering_limit_left"]) / CAR["wheelbase"]
KAPPA_RIGHT = math.tan(CAR["steering_limit_right"]) / CAR["wheelbase"]
ARROWS = {'39': 'LEFT', 'LEFT': 'LEFT', '38': 'RIGHT', 'RIGHT': 'RIGHT'}   # YOLO id -> side

# The carpark (task2_arena.sdf), drawn in Foxglove with the blocks: centre, size.
CARPARK = ((0.0, 0.0, 0.001), (0.6, 0.5, 0.002))


class State(Enum):
    WAITING_FOR_START = auto()
    DRIVING_PATH = auto()
    FINISHED = auto()


def slalom_waypoints(arrow: str, o1, o2):
    """Pass obstacle 1 on the arrow's side, obstacle 2 on the other, loop round
    behind obstacle 2 and come back into the carpark (at the origin). Placed
    from the blocks' centres (o1, o2: obstacle_layout.LayoutObstacle)."""
    side1 = 1.0 if arrow == 'LEFT' else -1.0
    side2 = -side1
    return [(0.35, 0.0),                               # out of the carpark
            (o1.x + 0.15, o1.y + 0.50 * side1),        # past obstacle 1
            ((o1.x + o2.x) / 2.0, o2.y),               # cross between the obstacles
            (o2.x - 0.15, o2.y + 0.50 * side2),        # before obstacle 2, other side
            (o2.x + 0.15, o2.y + 0.50 * side2),        # past obstacle 2
            (o2.x + 0.40, o2.y),                       # behind obstacle 2
            (o2.x + 0.15, o2.y - 0.50 * side2),        # back along its other side
            (o1.x, o1.y - 0.20 * side2),               # onto the return line
            (0.00, 0.00)]                              # into the carpark


class Task2Runner(Node):
    def __init__(self):
        super().__init__('task2_runner')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        # fast_speed, slow_speed, lookahead_dist (live): config/navigation.yaml.
        config.declare(self, 'task2_runner')
        layout = self.declare_parameter(
            'layout', f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml").value
        self.obstacles = obstacle_layout.load(layout, 'task2')
        if len(self.obstacles) != 2:
            raise ValueError(f'{layout}: task2 needs exactly 2 obstacles, found {len(self.obstacles)}')
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        self.create_subscription(String, '/yolo_result', self.arrow_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_service(Trigger, '/start_run', self.start_run_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        wall_timer(self, 0.05, self.control_loop)   # 20 Hz

        self.planner = SplinePathPlanner()
        self.state = State.WAITING_FOR_START
        self.go_received = False
        self.current_pose = (0.0, 0.0, 0.0)   # carpark at the arena origin, facing +x
        self.arrow1, self.arrow2 = 'LEFT', 'RIGHT'
        self.waypoints = []
        self.dense_path = []
        self.path_idx = 0
        self.current_wpt_idx = 0
        self.get_logger().info("Task 2 Runner Initialized! Holding - waiting for `go` (/start_run service)...")

    def arrow_callback(self, msg: String):
        side = ARROWS.get(msg.data.strip().upper())
        if side is None:
            return                     # not an arrow
        # YOLO reports every frame: log only a change.
        if self.state == State.WAITING_FOR_START or self.current_wpt_idx < 3:
            if side != self.arrow1:
                self.get_logger().info(f"Arrow 1 set to: {side}")
            self.arrow1 = side
        else:
            if side != self.arrow2:
                self.get_logger().info(f"Arrow 2 set to: {side}")
            self.arrow2 = side

    def start_run_callback(self, request, response):
        """/start_run (`pixi run go`): only while waiting to start."""
        response.success = self.state == State.WAITING_FOR_START
        if response.success:
            self.go_received = True
            response.message = "Run started - driving slalom path."
            self.get_logger().info("Received `go` - starting Task 2 path.")
        else:
            response.message = f"Ignored: run already in progress ({self.state.name})."
            self.get_logger().warn(f"`go` rejected: {response.message}")
        return response

    def odom_callback(self, msg: Odometry) -> None:
        """EKF pose (odom) -> map via TF; on a lookup failure keep the last pose."""
        try:
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"No map <- {msg.header.frame_id} transform yet ({exc}); "
                                   f"keeping previous pose {self.current_pose}", throttle_duration_sec=2.0)
            return
        pose = tf2_geometry_msgs.do_transform_pose(msg.pose.pose, tf)
        self.current_pose = (pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation))
        self.pose_pub.publish(PoseStamped(header=markers.header(msg.header.stamp), pose=pose))

    def send_cmd(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(linear_x), float(angular_z)
        self.cmd_pub.publish(msg)

    def publish_visualizations(self):
        if not self.waypoints:
            return
        stamp = self.get_clock().now().to_msg()
        path = self.dense_path or [(x, y, 0.0) for x, y in self.waypoints]
        self.path_pub.publish(markers.route_path([path], stamp))
        items = [markers.marker(stamp, 'dense_path', 0, Marker.LINE_STRIP, (0.1, 0.6, 1.0, 0.9),
                                scale=(0.02, 1, 1), points=[(x, y, 0.03) for x, y, _ in path])]
        items += [markers.marker(stamp, 'arena', i, Marker.CUBE, (0.2, 0.2, 0.2, 0.9),
                                 o.x, o.y, o.size[2] / 2.0, scale=o.size)
                  for i, o in enumerate(self.obstacles)]
        items.append(markers.marker(stamp, 'arena', len(self.obstacles), Marker.CUBE,
                                    (0.2, 0.8, 0.2, 0.4), *CARPARK[0], scale=CARPARK[1]))
        for i, (x, y) in enumerate(self.waypoints):
            rgba = (0.0, 1.0, 0.0, 0.9) if i == self.current_wpt_idx else (1.0, 1.0, 0.0, 0.9)
            items += [markers.marker(stamp, 'waypoints', i, Marker.SPHERE, rgba, x, y, 0.05,
                                     scale=(0.12, 0.12, 0.12)),
                      markers.marker(stamp, 'waypoint_labels', 100 + i, Marker.TEXT_VIEW_FACING,
                                     (1.0, 1.0, 1.0, 1.0), x, y, 0.20, scale=(1, 1, 0.10),
                                     text=f"W{i} ({x:.2f}, {y:.2f})")]
        self.marker_pub.publish(MarkerArray(markers=items))

    def lookahead_index(self) -> int:
        """Index of the first path point at least lookahead_dist away. The search
        start only moves forward (never chases points already passed)."""
        lookahead = float(self.get_parameter('lookahead_dist').value)
        x, y, _ = self.current_pose
        n = len(self.dense_path)
        while self.path_idx < n - 1 and math.hypot(self.dense_path[self.path_idx][0] - x,
                                                   self.dense_path[self.path_idx][1] - y) < lookahead * 0.5:
            self.path_idx += 1
        for i in range(self.path_idx, n):
            if math.hypot(self.dense_path[i][0] - x, self.dense_path[i][1] - y) >= lookahead:
                return i
        return n - 1

    def control_loop(self):
        if self.state == State.WAITING_FOR_START:
            self.send_cmd(0.0, 0.0)
            if self.go_received:
                self.start()
        elif self.state == State.DRIVING_PATH:
            self.publish_visualizations()
            self.drive()
        else:
            self.publish_visualizations()
            self.send_cmd(0.0, 0.0)

    def start(self):
        self.waypoints = slalom_waypoints(self.arrow1, *self.obstacles)
        self.dense_path = self.planner.generate_path_through_waypoints(self.waypoints, step_size=0.05)
        peak = self.planner.max_curvature(self.dense_path)
        self.path_idx = self.current_wpt_idx = 0
        self.state = State.DRIVING_PATH
        feasible = "OK" if peak <= KAPPA_RIGHT else "EXCEEDS the right-turn limit!"
        self.get_logger().info(f"Starting Task 2 Execution! Planned {len(self.dense_path)} dense path points | "
                               f"peak curvature={peak:.2f} (1/m), right-turn limit={KAPPA_RIGHT:.2f} [{feasible}]")

    def drive(self):
        x, y, yaw = self.current_pose
        end_x, end_y, _ = self.dense_path[-1]
        if self.path_idx >= len(self.dense_path) - 1 and math.hypot(end_x - x, end_y - y) < 0.10:
            self.send_cmd(0.0, 0.0)
            self.state = State.FINISHED
            self.get_logger().info("Task 2 Slalom Path Completed! Stopped in Carpark.")
            return

        tx, ty, _ = self.dense_path[self.lookahead_index()]
        local_x = (tx - x) * math.cos(yaw) + (ty - y) * math.sin(yaw)
        local_y = -(tx - x) * math.sin(yaw) + (ty - y) * math.cos(yaw)
        curvature = 2.0 * local_y / max(math.hypot(local_x, local_y), 0.05) ** 2
        curvature = max(-KAPPA_RIGHT, min(KAPPA_LEFT, curvature))   # what the steering can do
        limit = KAPPA_LEFT if curvature >= 0 else KAPPA_RIGHT
        speed = float(self.get_parameter('slow_speed' if abs(curvature) > 0.5 * limit else 'fast_speed').value)
        self.send_cmd(speed, speed * curvature)

        if self.current_wpt_idx < len(self.waypoints):   # progress log only
            wx, wy = self.waypoints[self.current_wpt_idx]
            if math.hypot(wx - x, wy - y) < 0.15:
                self.get_logger().info(f"[Wpt {self.current_wpt_idx}] Target({wx:.2f}, {wy:.2f}) | "
                                       f"Pose({x:.2f}, {y:.2f})")
                self.current_wpt_idx += 1


def main(args=None):
    run(Task2Runner, args=args)


if __name__ == '__main__':
    main()
