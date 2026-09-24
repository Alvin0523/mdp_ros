#!/usr/bin/env python3
"""Always-on base node: position feedback to the tablet and pose reset.

`/reset_pose` (Trigger, `pixi run reset`) puts the EKF pose back at the start pose.

Turns the EKF pose into `ROBOT,<x>,<y>,<N/E/S/W>` (cell x = column, y = row from
the arena's bottom-left, 10 cm cells, 0..19) and publishes it on /bluetooth_tx.
Sent when it changes and repeated every 2 s, so a tablet (or bridge) that comes
up later still learns the pose. Part of the base bringup, independent of task
runners.
"""
import math

import rclpy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(String, '/bluetooth_tx', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)
        # `odom` starts at the start pose (map -> odom is the static start-pose
        # transform), so "back at the start" means EKF pose = identity in odom.
        self.set_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/set_pose', 10)
        self.create_service(Trigger, '/reset_pose', self.reset_pose)
        self.last_line = None
        self.create_timer(2.0, self.heartbeat)

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
            self.get_logger().warn(f'No {self.arena_frame} transform yet ({exc})',
                                   throttle_duration_sec=5.0)
            return
        pose = tf2_geometry_msgs.do_transform_pose(pose_in.pose, tf)
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        line = robot_cell_line(pose.position.x, pose.position.y, yaw)
        if line != self.last_line:
            self.last_line = line
            self.send(line)

    def reset_pose(self, request, response):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.orientation.w = 1.0
        for i in range(0, 36, 7):
            msg.pose.covariance[i] = 1e-6
        self.set_pose_pub.publish(msg)
        self.get_logger().info('Pose reset to the start pose.')
        response.success = True
        response.message = 'EKF pose reset to the start pose.'
        return response

    def heartbeat(self) -> None:
        if self.last_line is not None:
            self.send(self.last_line)


def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseFeedback()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
