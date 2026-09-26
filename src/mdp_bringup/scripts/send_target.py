#!/usr/bin/env python3
"""Debug: send `TARGET,<obstacle number>,<target id>` to the tablet by hand.

    pixi run target 2 20            ->  TARGET,2,20

Publishes on /bluetooth_tx, the same topic the bridge forwards to the tablet, so it
shows on /bt_log too. Note the bridge drops a line identical to the one it sent last,
so to resend the same TARGET, send a different one in between.
Both values must be non-negative whole numbers.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    if len(argv) != 2:
        print('usage: send_target.py <obstacle number> <target id>   e.g. send_target.py 2 20',
              file=sys.stderr)
        sys.exit(2)
    values = []
    for name, text in (('obstacle number', argv[0]), ('target id', argv[1])):
        try:
            value = int(text)
        except ValueError:
            print(f'{name} must be a whole number, got {text!r}', file=sys.stderr)
            sys.exit(2)
        if value < 0:
            print(f'{name} must not be negative, got {value}', file=sys.stderr)
            sys.exit(2)
        values.append(value)
    line = f'TARGET,{values[0]},{values[1]}'
    rclpy.init()
    node = Node('send_target')
    pub = node.create_publisher(String, '/bluetooth_tx', 10)
    # /bluetooth_tx is volatile: wait until the bridge is subscribed or it is lost.
    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        node.get_logger().error('nobody subscribed to /bluetooth_tx - is the bringup running?')
        sys.exit(1)
    pub.publish(String(data=line))
    rclpy.spin_once(node, timeout_sec=0.3)
    node.get_logger().info(f'sent {line}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
