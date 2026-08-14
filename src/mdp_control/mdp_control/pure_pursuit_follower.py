#!/usr/bin/env python3
"""
Pure Pursuit Path Follower for Ackermann Steering Controller.
Calculates steering angle (angular.z) and velocity (linear.x) to track path waypoints.
"""

import math
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState

class PurePursuitFollower(Node):
    def __init__(self):
        super().__init__('pure_pursuit_follower')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_cb, 10)
        
        # Vehicle & Pursuer Kinematics
        self.wheelbase = 0.147          # L = 0.147 m
        self.lookahead_dist = 0.25       # Lookahead distance 25 cm
        self.max_steering_angle = 0.39   # Max steering angle ~22.35 deg
        self.target_speed = 0.5          # Default target linear velocity (m/s)

        # Path & Pose State
        self.path: List[Tuple[float, float, float]] = []
        self.current_pose = [0.0, 0.0, 0.0]  # x, y, yaw
        self.active = False

    def joint_states_cb(self, msg: JointState):
        pass

    def set_path(self, path_waypoints: List[Tuple[float, float, float]]):
        self.path = path_waypoints
        self.active = True

    def calculate_pure_pursuit(self, target_point: Tuple[float, float]) -> float:
        dx = target_point[0] - self.current_pose[0]
        dy = target_point[1] - self.current_pose[1]
        
        yaw = self.current_pose[2]
        local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        
        if local_x <= 0:
            return 0.0

        curvature = (2.0 * local_y) / (self.lookahead_dist ** 2)
        steering_angle = math.atan(self.wheelbase * curvature)
        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
        return steering_angle

    def publish_cmd(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)
        self.ref_pub.publish(msg)

    def stop(self):
        self.active = False
        self.publish_cmd(0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
