#!/usr/bin/env python3
"""
One-shot test-obstacle publisher for task1_runner.py's /obstacle_setup.

In the real competition, obstacle positions/facings come from the Android
tablet over Bluetooth (see docs/rpi/algorithm.md / assessment_checklist.md's
"Preparation" step - the tablet receives them from the supervisor and
forwards to the RPi). During development, there's no tablet in the loop -
this reads a small YAML file instead and publishes it in the exact
pipe-delimited format task1_runner.py's setup_callback() parses
("id:x,y,facing|id:x,y,facing|..."), so the rest of the pipeline (planner,
visualization) can be exercised without waiting on the Bluetooth/tablet
integration to exist.

Publishes repeatedly for a few seconds (not a durable/transient-local
publish) since /obstacle_setup's subscriber uses default (volatile) QoS -
a single publish immediately at node startup could race ahead of
task1_runner's subscription being established, especially if both are
launched together.
"""
import sys

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

DEFAULT_CONFIG = f"{get_package_share_directory('mdp_bringup')}/config/test_obstacles.yaml"


def format_obstacle_setup(obstacles: list) -> str:
    """obstacles: list of {id, x, y, facing} dicts (from YAML) -> the
    pipe-delimited wire format task1_runner.setup_callback() expects."""
    items = []
    for obs in obstacles:
        items.append(f"{obs['id']}:{obs['x']},{obs['y']},{obs['facing']}")
    return '|'.join(items)


class TestObstaclePublisher(Node):
    def __init__(self, config_path: str):
        super().__init__('publish_test_obstacles')

        with open(config_path) as f:
            data = yaml.safe_load(f)
        obstacles = data.get('obstacles', [])
        if not obstacles:
            self.get_logger().warn(f"No obstacles found in {config_path} - nothing to publish.")

        self.message = format_obstacle_setup(obstacles)
        self.get_logger().info(f"Loaded {len(obstacles)} obstacles from {config_path}")
        self.get_logger().info(f"Publishing: {self.message}")

        self.pub = self.create_publisher(String, '/obstacle_setup', 10)
        self.publish_count = 0
        # Every 0.5s for 5s (10 publishes) - task1_runner only ever acts on
        # the first one it receives (setup_callback() checks
        # State.WAITING_FOR_SETUP), so republishing is harmless, just a
        # safety margin against the subscription not being up yet.
        self.timer = self.create_timer(0.5, self.publish_once)

    def publish_once(self):
        msg = String()
        msg.data = self.message
        self.pub.publish(msg)
        self.publish_count += 1
        if self.publish_count >= 10:
            self.get_logger().info("Done publishing test obstacles.")
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
