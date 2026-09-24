#!/usr/bin/env python3
"""Live view of the tablet's Bluetooth link, in one terminal (`pixi run bt`).

    LINK UP / LINK DOWN        /bluetooth_bridge/link_ok  (only printed on change)
    TABLET -> RPI   <line>     /bluetooth_rx   (what the tablet sent)
    RPI -> TABLET   <line>     /bluetooth_tx   (what the Pi sends it)

Needs the bluetooth_bridge_node running (any `pixi run real ...` starts it).
The runner resends ROBOT/PLAN/RESET/STATUS every 2 s, so use --changes to hide
repeated identical lines (default), or --all to see every one.
"""
import argparse
import time

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from std_msgs.msg import Bool, String


class BtMonitor(Node):
    def __init__(self, show_all: bool, quiet: bool = False):
        super().__init__('bt_monitor')
        self.log_pub = self.create_publisher(Log, '/bt_log', 50)
        self.show_all = show_all
        self.quiet = quiet
        self.last_tx = {}       # line kind (text before ':' or ',') -> last line
        self.link = None
        self.create_subscription(Bool, '/bluetooth_bridge/link_ok', self.on_link, 10)
        self.create_subscription(String, '/bluetooth_rx', self.on_rx, 50)
        self.create_subscription(String, '/bluetooth_tx', self.on_tx, 50)
        self.get_logger().info('watching /bluetooth_bridge/link_ok, /bluetooth_rx, /bluetooth_tx')

    @staticmethod
    def stamp() -> str:
        return time.strftime('%H:%M:%S')

    def show(self, text: str):
        # Terminal line, plus the same text on /bt_log (rcl_interfaces/Log) so
        # Foxglove's Log panel shows a scrolling history.
        if not self.quiet:
            print(f'{self.stamp()}  {text}', flush=True)
        m = Log()
        m.stamp = self.get_clock().now().to_msg()
        m.level = Log.INFO
        m.name = 'bt'
        m.msg = text
        self.log_pub.publish(m)

    def on_link(self, msg: Bool):
        if msg.data != self.link:
            self.link = msg.data
            self.show(f'LINK {"UP" if msg.data else "DOWN"}')

    def on_rx(self, msg: String):
        self.show(f'TABLET -> RPI   {msg.data}')

    def on_tx(self, msg: String):
        line = msg.data
        kind = line.replace(',', ':').split(':')[0]
        if not self.show_all and self.last_tx.get(kind) == line:
            return
        self.last_tx[kind] = line
        # The runner publishes these whether or not a tablet is connected; the
        # bridge only writes them to the device while the link is up (and resends
        # the latest ROBOT/PLAN/RESET/STATUS when it comes up).
        note = '' if self.link else '   [link down - not delivered yet]'
        self.show(f'RPI -> TABLET   {line}{note}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--all', action='store_true',
                    help='print every /bluetooth_tx line, including the 2 s repeats')
    ap.add_argument('--quiet', action='store_true',
                    help='do not print to the terminal, only publish /bt_log (used by the launch)')
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = BtMonitor(args.all, args.quiet)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
