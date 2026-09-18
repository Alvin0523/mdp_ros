#!/usr/bin/env python3
"""Approach an obstacle on the front ultrasonic, then drive one full square
loop around it so the left-facing camera sees all 4 faces for YOLO.

WHY THIS EXISTS: neither drive_distance.py nor rotate_angle.py know about the
obstacle - they are the building blocks (closed-loop straight leg, closed-loop
turn), this script is the behaviour on top. It does two things:

1. APPROACH - drive forward watching /ultrasonic (front-facing HC-SR04) and
   stop once the obstacle is within --stop-cm. This phase is its own small
   node because it is the only part that needs the ultrasonic reading;
   nothing else here does.

2. LOOP - the camera faces LEFT (90 deg off chassis-forward), not forward, so
   stopping square-on to the obstacle shows the camera nothing useful - it is
   looking out to the side, past the obstacle. To get all 4 faces in view,
   drive a full square around it: turn --turn-deg (default 90) the SAME
   direction 4 times, with a straight --leg-m leg driven between each turn.
   Shelling out to rotate_angle.py and drive_distance.py for this reuses
   their closed-loop stopping (EKF yaw / wheel odometry) instead of
   duplicating it here.

   Direction defaults to LEFT. Picture driving laps on an oval track: turning
   left the whole way round keeps the infield on the driver's left the whole
   time. Same here - the obstacle is the "infield", the camera looks left, so
   a left-turning (CCW, viewed from above) loop keeps the obstacle on the
   camera's side for the entire lap. --direction right sweeps the other way,
   for a chassis where that mapping is flipped.

Usage:
    ros2 run mdp_bringup circle_obstacle.py
    ros2 run mdp_bringup circle_obstacle.py --stop-cm 30 --leg-m 0.6
    ros2 run mdp_bringup circle_obstacle.py --direction right
    pixi run circle
"""

import argparse
import math
import subprocess
import sys

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Range

CMD_TOPIC = '/cmd_vel'
ULTRASONIC_TOPIC = '/ultrasonic'
PUBLISH_HZ = 20.0


class ApproachObstacle(Node):
    """Drive straight forward until the front ultrasonic reports the
    obstacle within stop_m, or give up after timeout_s with nothing seen."""

    def __init__(self, stop_m: float, speed_mps: float, timeout_s: float):
        super().__init__('approach_obstacle')

        self.stop_m = stop_m
        self.speed_mps = speed_mps
        self.timeout_s = timeout_s

        self.last_range_m = math.inf
        self.have_reading = False
        self.finished = False
        self.reached = False
        self.start_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, 10)
        self.create_subscription(
            Range, ULTRASONIC_TOPIC, self._on_range,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        self.get_logger().info(
            f'approaching obstacle, stopping at {self.stop_m * 100:.0f} cm, '
            f'watching {ULTRASONIC_TOPIC}')

    def _on_range(self, msg: Range) -> None:
        self.last_range_m = msg.range
        self.have_reading = True

    def _publish(self, vx: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx
        self.cmd_pub.publish(msg)

    def _tick(self) -> None:
        if self.finished:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if not self.have_reading:
            self._publish(0.0)
            if elapsed > 5.0:
                self._stop(False, f'no reading on {ULTRASONIC_TOPIC} after 5s - '
                           f'is bringup running?')
            return

        if math.isfinite(self.last_range_m) and self.last_range_m <= self.stop_m:
            self._stop(True, f'obstacle at {self.last_range_m * 100:.1f} cm '
                       f'(threshold {self.stop_m * 100:.0f} cm)')
            return

        if elapsed > self.timeout_s:
            self._stop(False, f'TIMEOUT after {elapsed:.1f}s - nothing within '
                       f'{self.stop_m * 100:.0f} cm (last reading '
                       f'{self.last_range_m * 100:.1f} cm)')
            return

        self._publish(self.speed_mps)

    def _stop(self, reached: bool, reason: str) -> None:
        self.finished = True
        self.reached = reached
        for _ in range(5):
            self._publish(0.0)
        self.get_logger().info(reason)


def approach(stop_cm: float, speed_mps: float, timeout_s: float) -> bool:
    node = ApproachObstacle(stop_cm / 100.0, speed_mps, timeout_s)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        return node.reached
    finally:
        node.destroy_node()


def run_step(argv: list) -> bool:
    print(f'>>> {" ".join(argv)}', flush=True)
    result = subprocess.run(argv)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Approach an obstacle on the front ultrasonic, then '
                    'drive one square loop around it (camera faces left).')
    parser.add_argument('--stop-cm', type=float, default=25.0,
                        help='stop the approach once the obstacle is this '
                             'close, cm, default 25')
    parser.add_argument('--leg-m', type=float, default=0.5,
                        help='length of each of the 4 sides of the loop, '
                             'metres, default 0.5 (must clear the obstacle '
                             'corner to corner)')
    parser.add_argument('--turn-deg', type=float, default=90.0,
                        help='turn angle at each corner, default 90')
    parser.add_argument('--direction', type=str, default='left',
                        choices=['left', 'right'],
                        help='turn direction for all 4 corners, default left '
                             '(keeps the obstacle on the left-facing camera '
                             'side for the whole loop)')
    parser.add_argument('--approach-speed', type=float, default=0.12,
                        help='m/s while approaching, default 0.12')
    parser.add_argument('--approach-timeout', type=float, default=20.0,
                        help='seconds to give up approaching if nothing is '
                             'seen, default 20')
    parser.add_argument('--drive-speed', type=float, default=0.15,
                        help='m/s for each straight leg, default 0.15')
    parser.add_argument('--turn-speed', type=float, default=0.15,
                        help='m/s while turning, default 0.15')
    parser.add_argument('--skip-approach', action='store_true',
                        help='skip the ultrasonic approach phase and start '
                             'the loop immediately (robot already in place)')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    try:
        if not args.skip_approach:
            reached = approach(args.stop_cm, args.approach_speed,
                                args.approach_timeout)
            if not reached:
                print('!!! approach did not reach the obstacle, aborting loop',
                      file=sys.stderr)
                return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    for corner in range(1, 5):
        print(f'--- corner {corner}/4: turn {args.turn_deg:.0f} deg '
              f'{args.direction} ---', flush=True)
        ok = run_step([
            'ros2', 'run', 'mdp_bringup', 'rotate_angle.py',
            str(args.turn_deg), '--direction', args.direction,
            '--speed', str(args.turn_speed),
        ])
        if not ok:
            print(f'!!! turn {corner} failed, aborting loop', file=sys.stderr)
            return 1

        print(f'--- corner {corner}/4: leg {args.leg_m:.2f} m ---', flush=True)
        ok = run_step([
            'ros2', 'run', 'mdp_bringup', 'drive_distance.py',
            str(args.leg_m), '--speed', str(args.drive_speed),
        ])
        if not ok:
            print(f'!!! leg {corner} failed, aborting loop', file=sys.stderr)
            return 1

    print('>>> DONE - one full loop around the obstacle complete', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
