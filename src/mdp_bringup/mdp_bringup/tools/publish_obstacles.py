"""
One-shot obstacle-setup publisher for task1_runner's /obstacle_setup, run by
mdp.launch.py at startup for obstacles:=yaml (the sim default). By hand, use
`pixi run setup` / Foxglove SETUP (task1_runner's /setup_obstacles) instead.

Reads the task 1 layout (config/tasks.yaml by default, see
obstacle_layout.py) and publishes exactly the message bluetooth_bridge_node
would publish after the tablet sent that set and DONE - cell centres, metres.
Useful without the tablet; mdp.launch.py also runs it at startup for
obstacles:=yaml (the sim default).

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

from mdp_bringup.utils import obstacle_layout

WAIT_FOR_SUBSCRIBER_S = 60.0
DEFAULT_CONFIG = f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml"


class ObstaclePublisher(Node):
    def __init__(self, config_path: str, block=None):
        super().__init__('publish_obstacles')

        if block is not None:
            # Blocks from the command line (pixi run block COL ROW FACE [COL ROW FACE ...]):
            # a short task 1 - e.g. testing the IR position fix at one or two checkpoints.
            obstacles = [obstacle_layout.LayoutObstacle(
                i + 1, (col + 0.5) * obstacle_layout.CELL_M, (row + 0.5) * obstacle_layout.CELL_M, facing)
                for i, (col, row, facing) in enumerate(block)]
            config_path = 'the command line'
        else:
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
        # Wait for task1_runner itself: sim_helpers subscribes too (sim), and
        # publishing as soon as IT was seen sometimes lost the set for the
        # runner (sim, 2026-10-01).
        if any(i.node_name == 'task1_runner'
               for i in self.get_subscriptions_info_by_topic(self.pub.topic_name)):
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
        self.get_logger().info("Published the obstacles once.")
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
    block = None
    if argv and argv[0] == '--block':
        # --block COL ROW N|E|S|W [COL ROW N|E|S|W ...] : blocks in tablet cells
        usage = 'usage: pixi run block COL ROW N|E|S|W [COL ROW N|E|S|W ...]   (cells 0..19, the image side)'
        args = argv[1:]
        if not args or len(args) % 3:
            sys.exit(usage)
        block = []
        for i in range(0, len(args), 3):
            try:
                col, row, facing = int(args[i]), int(args[i + 1]), args[i + 2].upper()
            except ValueError:
                sys.exit(usage)
            if facing not in ('N', 'E', 'S', 'W') or not (0 <= col < 20 and 0 <= row < 20):
                sys.exit(usage)
            block.append((col, row, facing))
        argv = []
    config_path = argv[0] if argv else DEFAULT_CONFIG
    node = ObstaclePublisher(config_path, block)
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
