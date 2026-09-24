#!/usr/bin/env python3
"""
TEMPORARY safety guard: stop the car once it leaves the tablet's 20x20 grid.

    ros2 run mdp_bringup grid_guard.py        (pixi run guard)

Rule: the ROBOT,<x>,<y> cell sent to the tablet must stay in 0..19 on both
axes. Same cell maths as task1_runner.robot_cell_line(), but computed WITHOUT
its clamp (the runner clamps to 0..18, so the tablet line itself never goes
past the edge). If x or y is over 19 (or below 0):

  1. calls /stop_run once, so the runner stops driving and holds zeros itself
     (otherwise its 20Hz /cmd_vel stream would just overwrite ours), and
  2. streams zero /cmd_vel at 50Hz for as long as the robot stays outside.

Back inside the grid it stops publishing. The runner stays STOPPED until
`pixi run reset`. Run alongside `pixi run real1` / `real2` / `drive`.
"""
import math

import rclpy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 - registers the PoseStamped transform
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger

# Must match task1_runner.py's tablet constants.
TABLET_CELL_CM = 10.0
ROBOT_FOOTPRINT_CM = 20.0
MAX_CELL = 19  # ROBOT x or y above this (or below 0) = off the grid -> stop

ZERO_RATE_HZ = 50.0


def raw_cell(v_m: float) -> int:
    """robot_cell_line()'s cell(), minus the clamp."""
    return int(math.floor((v_m * 100.0 - ROBOT_FOOTPRINT_CM / 2.0) / TABLET_CELL_CM))


class GridGuard(Node):
    def __init__(self):
        super().__init__('grid_guard')
        self.declare_parameter('arena_frame', 'map')
        self.declare_parameter('call_stop_run', True)
        self.arena_frame = self.get_parameter('arena_frame').value
        self.call_stop_run = self.get_parameter('call_stop_run').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.stop_client = self.create_client(Trigger, '/stop_run')
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_timer(1.0 / ZERO_RATE_HZ, self.hold_zero)

        self.outside = False
        self.stop_sent = False
        self.get_logger().info(
            f"Grid guard up: stopping if the ROBOT cell leaves 0..{MAX_CELL}.")

    def odom_callback(self, msg: Odometry) -> None:
        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.pose = msg.pose.pose
        try:
            tf = self.tf_buffer.lookup_transform(
                self.arena_frame, msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                f"No {self.arena_frame} <- {msg.header.frame_id} transform yet ({exc})",
                throttle_duration_sec=2.0)
            return
        p = tf2_geometry_msgs.do_transform_pose(pose_in.pose, tf).position
        cx, cy = raw_cell(p.x), raw_cell(p.y)
        outside = not (0 <= cx <= MAX_CELL and 0 <= cy <= MAX_CELL)

        if outside and not self.outside:
            self.get_logger().error(
                f"OUT OF GRID: cell ({cx},{cy}) at ({p.x:.2f}, {p.y:.2f}) m - holding zero /cmd_vel.")
            self.send_stop_run()
        elif not outside and self.outside:
            self.get_logger().info(f"Back inside the grid at cell ({cx},{cy}).")
            self.stop_sent = False
        self.outside = outside

    def send_stop_run(self) -> None:
        if not self.call_stop_run or self.stop_sent:
            return
        if not self.stop_client.service_is_ready():
            self.get_logger().warn("/stop_run not available - only zeroing /cmd_vel.")
            return
        self.stop_client.call_async(Trigger.Request())
        self.stop_sent = True

    def hold_zero(self) -> None:
        if not self.outside:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GridGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
