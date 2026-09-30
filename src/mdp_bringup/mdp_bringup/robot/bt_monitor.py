"""Live view of the tablet's Bluetooth link: `ros2 run mdp_bringup bt_monitor`.

    LINK UP / LINK DOWN        /bluetooth_bridge/link_ok  (only printed on change)
    TABLET -> RPI   <line>     /bluetooth_rx   (what the tablet sent)
    RPI -> TABLET   <line>     /bluetooth_tx   (what the Pi sends it)

Each is a normal log line on /rosout from node `bt_monitor` (`pixi run btlog`,
Foxglove Log panel). Needs the bluetooth_bridge_node running (any launch starts
it); the launch runs this too, kept off its terminal. ROBOT/PLAN/RESET/STATUS
are resent every 2 s; repeats are hidden unless --all.
"""
import argparse

from rclpy.node import Node
from std_msgs.msg import Bool, String
from mdp_bringup.utils.run import run


class BtMonitor(Node):
    def __init__(self, show_all: bool):
        super().__init__('bt_monitor')
        self.show_all = show_all
        self.last_tx = {}       # line kind (text before ':' or ',') -> last line
        self.link = None
        self.create_subscription(Bool, '/bluetooth_bridge/link_ok', self.on_link, 10)
        self.create_subscription(String, '/bluetooth_rx', self.on_rx, 50)
        self.create_subscription(String, '/bluetooth_tx', self.on_tx, 50)
        self.get_logger().info('watching /bluetooth_bridge/link_ok, /bluetooth_rx, /bluetooth_tx')

    def show(self, text: str):
        self.get_logger().info(text)

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
    args, ros_args = ap.parse_known_args()
    run(lambda: BtMonitor(args.all), args=ros_args)


if __name__ == '__main__':
    main()
