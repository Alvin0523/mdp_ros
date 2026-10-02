"""Always-on base node: position feedback to the tablet and pose reset.

`/reset_pose` (Trigger, `pixi run reset`) puts the EKF pose back at the start pose.
On the real car the serial bridge first re-measures the gyro bias over 2 s
(/hardware_bridge/zero_gyro) - keep the car still - and the pose is reset once,
when it reports (/hardware_bridge/gyro_bias), so the heading starts clean. It
used to reset at once AND again after the bias: two RESETs on the tablet.

Turns the EKF pose into `ROBOT,<x>,<y>,<N/E/S/W>` (cell x = column, y = row from
the arena's bottom-left, 10 cm cells, 0..19) and publishes it on /bluetooth_tx.
Sent when it changes and repeated every 2 s, so a tablet (or bridge) that comes
up later still learns the pose. Part of the base bringup, independent of task
runners.
"""
import math
import subprocess
import time

import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from mdp_bringup.utils.run import run, wall_timer

CELL_CM = 10.0
MAX_CELL = 19


def robot_cell_line(x_m: float, y_m: float, yaw_rad: float) -> str:
    def cell(v_m: float) -> int:
        return max(0, min(MAX_CELL, int(math.floor(v_m * 100.0 / CELL_CM))))

    quarter = int(round(math.atan2(math.sin(yaw_rad), math.cos(yaw_rad)) / (math.pi / 2.0))) % 4
    return f'ROBOT,{cell(x_m)},{cell(y_m)},{"ENWS"[quarter]}'


class RobotPoseFeedback(Node):
    def __init__(self):
        super().__init__('robot_pose_feedback')
        self.arena_frame = self.declare_parameter('arena_frame', 'map').value
        # Sim only: the Gazebo world, to put the car itself back at the start on
        # /reset_pose (on the real car someone carries it back). '' = real car.
        self.gz_world = self.declare_parameter('gz_world', '').value
        self.start = tuple(self.declare_parameter(n, 0.0).value for n in ('start_x', 'start_y', 'start_yaw'))
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(String, '/bluetooth_tx', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)
        # `odom` starts at the start pose (map -> odom is the static start-pose
        # transform), so "back at the start" means EKF pose = identity in odom.
        self.set_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/set_pose', 10)
        self.create_service(Trigger, '/reset_pose', self.reset_pose)
        # Real car only (no serial bridge in sim): gyro bias re-measure on reset.
        self.zero_gyro = self.create_client(Trigger, '/hardware_bridge/zero_gyro')
        self.create_subscription(Float64, '/hardware_bridge/gyro_bias', self.on_gyro_bias, 10)
        self.bias_deadline = None    # s (wall): resetting once the gyro bias arrives, or then
        self.last_line = None
        wall_timer(self, 2.0, self.heartbeat)
        wall_timer(self, 0.5, self.check_bias_deadline)

    def send(self, line: str) -> None:
        self.pub.publish(String(data=line))

    def odom_cb(self, msg: Odometry) -> None:
        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.pose = msg.pose.pose
        try:
            tf = self.tf_buffer.lookup_transform(
                self.arena_frame, msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f'POSE      no {self.arena_frame} transform yet ({exc})',
                                   throttle_duration_sec=5.0)
            return
        pose = tf2_geometry_msgs.do_transform_pose(pose_in.pose, tf)
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        line = robot_cell_line(pose.position.x, pose.position.y, yaw)
        if line != self.last_line:
            self.last_line = line
            self.send(line)

    def teleport_to_start(self) -> bool:
        """Sim: move the Gazebo car back to the start pose (gz set_pose)."""
        x, y, yaw = self.start
        req = (f'name: "mini_akm_robot", position: {{x: {x}, y: {y}, z: 0.02}}, '
               f'orientation: {{z: {math.sin(yaw / 2.0)}, w: {math.cos(yaw / 2.0)}}}')
        out = subprocess.run(['gz', 'service', '-s', f'/world/{self.gz_world}/set_pose',
                              '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                              '--timeout', '2000', '--req', req], capture_output=True, text=True)
        ok = 'data: true' in out.stdout
        if not ok:
            self.get_logger().warn(f'RESET     Gazebo set_pose failed: {out.stdout.strip()} {out.stderr.strip()}')
        return ok

    def publish_start_pose(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.orientation.w = 1.0
        for i in range(0, 36, 7):
            msg.pose.covariance[i] = 1e-6
        self.set_pose_pub.publish(msg)

    def reset_pose(self, request, response):
        response.success = True
        if not self.gz_world and self.zero_gyro.service_is_ready():
            # Real car: the gyro first, then the pose (on_gyro_bias).
            self.zero_gyro.call_async(Trigger.Request())
            self.bias_deadline = time.monotonic() + 4.0
            response.message = 'Keep the car still: measuring the gyro bias (2 s), then the pose resets.'
            return response
        moved = self.teleport_to_start() if self.gz_world else False
        self.publish_start_pose()
        response.message = ('Car moved back to the start in Gazebo, EKF pose reset.' if moved
                            else 'EKF pose reset to the start pose.')
        return response

    def on_gyro_bias(self, msg: Float64) -> None:
        """The bridge measured the bias (or kept the old one, the car moved -
        it says so): reset the pose now, so nothing measured before counts."""
        if self.bias_deadline is not None:
            self.bias_deadline = None
            self.publish_start_pose()

    def check_bias_deadline(self) -> None:
        if self.bias_deadline is not None and time.monotonic() > self.bias_deadline:
            self.bias_deadline = None
            self.publish_start_pose()
            self.get_logger().warn('RESET     no gyro bias from the serial bridge - pose reset with the old bias')

    def heartbeat(self) -> None:
        if self.last_line is not None:
            self.send(self.last_line)


def main(args=None):
    run(RobotPoseFeedback, args=args)


if __name__ == '__main__':
    main()
