#!/usr/bin/env python3
"""
One-shot obstacle-setup publisher for task1_runner.py's /obstacle_setup
(`pixi run setup`).

Reads a layout YAML in tablet cells (config/test_obstacles.yaml by default, see
obstacle_layout.py) and publishes exactly the message bluetooth_bridge_node
would publish after the tablet sent that set and DONE - cell centres, metres.
Useful on the real robot without the tablet; the sim instead runs
fake_tablet.py, which goes through the bridge like the real tablet does.

Publishes exactly ONCE, like the bridge on DONE - task1_runner treats every
message as a new set and replans, so repeated publishes would restart planning
each time. /obstacle_setup's subscriber is volatile QoS, so a publish before
task1_runner has subscribed would be lost; the node waits for a subscriber
first (up to WAIT_FOR_SUBSCRIBER_S).
"""
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

import obstacle_layout

WAIT_FOR_SUBSCRIBER_S = 60.0
DEFAULT_CONFIG = f"{get_package_share_directory('mdp_bringup')}/config/test_obstacles.yaml"


class TestObstaclePublisher(Node):
    def __init__(self, config_path: str):
        super().__init__('publish_test_obstacles')

        obstacles = obstacle_layout.load(config_path)
        if not obstacles:
            self.get_logger().warn(f"No obstacles found in {config_path} - nothing to publish.")

        self.message = obstacle_layout.setup_string(obstacles)
        self.get_logger().info(f"Loaded {len(obstacles)} obstacles from {config_path}")
        self.get_logger().info(f"Publishing (cell x,y, facing): {obstacle_layout.describe(obstacles)}")

        self.pub = self.create_publisher(String, '/obstacle_setup', 10)
        self.waited = 0.0
        self.seen_subscriber = False
        self.timer = self.create_timer(0.5, self.publish_when_subscribed)

    def publish_when_subscribed(self):
        if self.pub.get_subscription_count() > 0:
            # Discovery can report the subscriber a moment before its
            # connection is ready to receive; give it one tick of grace.
            if not self.seen_subscriber:
                self.seen_subscriber = True
                return
        else:
            self.waited += 0.5
            if self.waited < WAIT_FOR_SUBSCRIBER_S:
                return
            self.get_logger().warn("No /obstacle_setup subscriber yet - publishing anyway.")
        msg = String()
        msg.data = self.message
        self.pub.publish(msg)
        self.get_logger().info("Published test obstacles once.")
        self.timer.cancel()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    # Drop ROS-injected args (--ros-args ...) that a launch Node action adds,
    # so the optional positional config path works both standalone and in a
    # launch file. First remaining token (if any) is the YAML path.
    argv = sys.argv[1:]
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    config_path = argv[0] if argv else DEFAULT_CONFIG
    node = TestObstaclePublisher(config_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
