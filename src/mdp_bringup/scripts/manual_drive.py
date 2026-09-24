#!/usr/bin/env python3
"""Tablet manual drive for the BARE car (`pixi run drive`, task:=0): no task runner.

Turns the tablet's drive letters on /manual_drive (f b fl fr bl br) into a short
burst on /cmd_vel, then a zero - the same behaviour and parameters as the task-1
runner's manual drive, so the tablet's buttons work with just `pixi run drive`.
Started by real.launch.py ONLY for task:=0: with task:=1 the runner already does
this, and two nodes reacting to the same letter would double the commands.

Publishes /cmd_vel only while a burst is active (plus one final zero), so it does
not fight dist / rotate / circle / teleop the way an always-on zero stream would.
Tune live:  ros2 param set /manual_drive manual_speed_mps 0.25
"""
import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import String

WHEELBASE_M = 0.1433
COMMANDS = {
    'f': (1.0, 0.0), 'b': (-1.0, 0.0),
    'fl': (1.0, 1.0), 'fr': (1.0, -1.0),
    'bl': (-1.0, 1.0), 'br': (-1.0, -1.0),
}


class ManualDrive(Node):
    def __init__(self):
        super().__init__('manual_drive')
        self.declare_parameter('manual_speed_mps', 0.15)
        self.declare_parameter('manual_steer_deg', 25.0)
        self.declare_parameter('manual_burst_s', 0.3)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(String, '/manual_drive', self.on_key, 10)
        self.create_timer(0.05, self.tick)
        self.cmd = (0.0, 0.0)
        self.until = 0.0
        self.active = False
        self.get_logger().info('manual drive ready: /manual_drive -> /cmd_vel (bare car)')

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_key(self, msg: String):
        key = msg.data.strip().lower()
        if key not in COMMANDS:
            self.get_logger().warn(f'unknown manual drive command {msg.data!r}')
            return
        direction, steer = COMMANDS[key]
        speed = min(0.5, max(0.05, float(self.get_parameter('manual_speed_mps').value)))
        steer_deg = min(30.0, max(5.0, float(self.get_parameter('manual_steer_deg').value)))
        burst = min(1.0, max(0.1, float(self.get_parameter('manual_burst_s').value)))
        v = direction * speed
        w = v * math.tan(math.radians(steer_deg)) / WHEELBASE_M * steer
        self.cmd = (v, w)
        self.until = self.now() + burst
        self.active = True

    def send(self, v: float, w: float):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.twist.linear.x = v
        m.twist.angular.z = w
        self.pub.publish(m)

    def tick(self):
        if self.until > self.now():
            self.send(*self.cmd)
        elif self.active:
            self.active = False
            self.send(0.0, 0.0)


def main():
    rclpy.init()
    node = ManualDrive()
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
