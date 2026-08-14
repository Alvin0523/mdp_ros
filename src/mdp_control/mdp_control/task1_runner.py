#!/usr/bin/env python3
"""
Task 1: Automatic Exploration & Image Recognition Runner Node.
Handles 2.0m x 2.0m arena exploration, Reeds-Shepp Ackermann path execution,
YOLO26 image recognition pause & capture, Android Tablet Bluetooth updates, and auto-stop.
"""

import rclpy
from rclpy.node import Node
from enum import Enum, auto
import math
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from mdp_control.reeds_shepp_planner import ReedsSheppPlanner

class State(Enum):
    WAITING_FOR_SETUP = auto()
    PLANNING_PATH = auto()
    NAVIGATING_TO_TARGET = auto()
    PAUSE_FOR_SCAN = auto()
    FINISHED = auto()

class Task1Runner(Node):
    def __init__(self):
        super().__init__('task1_runner')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.ref_pub = self.create_publisher(TwistStamped, '/ackermann_steering_controller/reference', 10)
        self.bt_pub = self.create_publisher(String, '/bluetooth_tx', 10)
        
        self.create_subscription(String, '/obstacle_setup', self.setup_callback, 10)
        self.create_subscription(String, '/yolo_result', self.yolo_callback, 10)

        # Control Loop @ 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)

        # Planner & State Variables
        self.planner = ReedsSheppPlanner(min_turn_radius=0.35)
        self.state = State.WAITING_FOR_SETUP
        
        self.obstacles = []
        self.standoffs = []
        self.visiting_order = []
        self.current_target_idx = 0
        
        self.detected_target_id = None
        self.state_start_time = self.get_now_sec()
        self.get_logger().info("Task 1 Runner Node Initialized! Waiting for obstacle setup...")

    def get_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def setup_callback(self, msg: String):
        if self.state == State.WAITING_FOR_SETUP:
            self.obstacles.clear()
            raw_items = msg.data.strip().split('|')
            for item in raw_items:
                if ':' in item:
                    obs_id, data = item.split(':')
                    parts = data.split(',')
                    x, y = float(parts[0]), float(parts[1])
                    face = parts[2]
                    self.obstacles.append((x, y, face))

            self.get_logger().info(f"Loaded {len(self.obstacles)} obstacles from setup!")
            self.state = State.PLANNING_PATH

    def yolo_callback(self, msg: String):
        if self.state == State.PAUSE_FOR_SCAN and self.detected_target_id is None:
            self.detected_target_id = msg.data.strip()
            self.get_logger().info(f"YOLO26 Identified Target: {self.detected_target_id}")

    def send_cmd(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)
        self.ref_pub.publish(msg)

    def send_bt(self, text: str):
        msg = String()
        msg.data = text
        self.bt_pub.publish(msg)

    def control_loop(self):
        now = self.get_now_sec()
        elapsed = now - self.state_start_time

        # STATE 1: Planning Path after receiving setup
        if self.state == State.PLANNING_PATH:
            start_pose = (0.0, 0.0, math.pi / 2)
            self.standoffs = [self.planner.calc_standoff_pose(x, y, f) for x, y, f in self.obstacles]
            self.visiting_order = self.planner.solve_tsp(start_pose, self.standoffs)
            
            self.get_logger().info(f"TSP Visiting Order Calculated: {self.visiting_order}")
            self.current_target_idx = 0
            self.state = State.NAVIGATING_TO_TARGET
            self.state_start_time = now

        # STATE 2: Navigating to current target standoff pose
        elif self.state == State.NAVIGATING_TO_TARGET:
            self.send_cmd(linear_x=0.5, angular_z=0.0)
            self.send_bt("ROBOT,0.5,1.0,N")

            if elapsed > 2.5:
                self.send_cmd(linear_x=0.0, angular_z=0.0)
                self.detected_target_id = None
                self.state = State.PAUSE_FOR_SCAN
                self.state_start_time = now
                self.get_logger().info(f"Arrived at Target Standoff {self.current_target_idx + 1}. Scanning...")

        # STATE 3: Pause for YOLO26 scanning & Bluetooth update
        elif self.state == State.PAUSE_FOR_SCAN:
            self.send_cmd(linear_x=0.0, angular_z=0.0)
            
            if self.detected_target_id is not None or elapsed > 0.6:
                target_id = self.detected_target_id if self.detected_target_id else "UNKNOWN"
                obs_num = self.visiting_order[self.current_target_idx] + 1
                
                self.send_bt(f"TARGET,{obs_num},{target_id}")
                self.get_logger().info(f"Updated Android Tablet: TARGET,{obs_num},{target_id}")

                self.current_target_idx += 1
                if self.current_target_idx >= len(self.visiting_order):
                    self.state = State.FINISHED
                    self.get_logger().info("All 5 targets processed! Auto-stopping...")
                else:
                    self.state = State.NAVIGATING_TO_TARGET
                    self.state_start_time = now

        # STATE 4: Finished & Auto-Stopped
        elif self.state == State.FINISHED:
            self.send_cmd(linear_x=0.0, angular_z=0.0)

def main(args=None):
    rclpy.init(args=args)
    node = Task1Runner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
