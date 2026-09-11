#!/usr/bin/env python3
"""Drive a straight line for a set distance, closed-loop on wheel odometry.

WHY THIS EXISTS: ROS has no "move 2 metres" command. /cmd_vel is a velocity,
so publishing it for a fixed duration is open-loop in time and depends on
battery voltage, load and how fast the PID settles. This node instead watches
odometry and stops when the distance is actually reported as covered.

PRIMARY USE - calibrating the distance scale. Wheel odometry is computed from
encoder ticks and wheel radius:

    metres_per_tick = 2*pi*wheel_radius / ticks_per_wheel_rev
                    = 2*pi*0.0325 / 1560  ~= 0.131 mm

Both of those constants came from WHEELTEC's reference material and neither has
been measured on this chassis. So run this with a target, then TAPE-MEASURE
what the robot actually travelled:

    correction = measured_distance / requested_distance

Anything other than ~1.0 means the constants are wrong by that factor. Note
odometry only depends on the PRODUCT 2*pi*r/ticks_per_rev, so this one test
gives the working correction even without knowing which of the two is off.
Wheel radius lives in mdp_bringup/config/real_controller.yaml
(traction_wheels_radius) and ticks-per-rev in mdp_stm32/src/motor.c
(MOTOR_TICKS_PER_REV).

Deliberately subscribes to the CONTROLLER's raw odometry rather than
/odometry/filtered: the EKF fuses IMU yaw rate on top, and for calibrating a
distance scale we want the quantity that is a direct function of the encoder
ticks and wheel radius, with nothing else mixed in.

Usage:
    ros2 run mdp_bringup drive_distance.py 2.0
    ros2 run mdp_bringup drive_distance.py 2.0 --speed 0.1
    ros2 run mdp_bringup drive_distance.py -1.0          # reverse

Requires the hardware bringup to be running (pixi run real1 / real2) and the
motor switch to be on.
"""

import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

CMD_TOPIC = '/cmd_vel'
ODOM_TOPIC = '/ackermann_steering_controller/odometry'
PUBLISH_HZ = 20.0

# Stop ramping down this far out so the car coasts onto the target rather than
# overshooting it - the drivetrain cannot stop instantly.
SLOWDOWN_MARGIN_M = 0.10
MIN_SPEED_MPS = 0.05


class DriveDistance(Node):
    def __init__(self, target_m: float, speed_mps: float, timeout_s: float):
        super().__init__('drive_distance')

        self.target_m = abs(target_m)
        self.direction = 1.0 if target_m >= 0.0 else -1.0
        self.speed_mps = abs(speed_mps)
        self.timeout_s = timeout_s

        self.start_xy = None
        self.travelled_m = 0.0
        self.finished = False
        self.start_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, 10)

        # Controller odometry is published best-effort in some configurations;
        # accepting either avoids a silent no-match on the subscription.
        self.create_subscription(
            Odometry, ODOM_TOPIC, self._on_odom,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        self.get_logger().info(
            f'target {self.target_m:.3f} m '
            f'{"forward" if self.direction > 0 else "reverse"} '
            f'at {self.speed_mps:.2f} m/s, odom from {ODOM_TOPIC}')

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.start_xy is None:
            self.start_xy = (x, y)
            return

        # Straight-line displacement from the start pose. Using displacement
        # rather than integrated path length keeps a slight curve from being
        # counted as extra forward progress.
        dx = x - self.start_xy[0]
        dy = y - self.start_xy[1]
        self.travelled_m = math.hypot(dx, dy)

    def _publish(self, vx: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def _tick(self) -> None:
        if self.finished:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed > self.timeout_s:
            self._stop(f'TIMEOUT after {elapsed:.1f}s - '
                       f'travelled {self.travelled_m:.3f} m of '
                       f'{self.target_m:.3f} m')
            return

        if self.start_xy is None:
            # No odometry yet - do not move. Publishing before the reference
            # pose is known would mean an unmeasured head start.
            self._publish(0.0)
            if elapsed > 5.0:
                self._stop(f'no odometry on {ODOM_TOPIC} after 5s - '
                           f'is the bringup running?')
            return

        remaining = self.target_m - self.travelled_m
        if remaining <= 0.0:
            self._stop(f'DONE - odometry reports {self.travelled_m:.3f} m '
                       f'(target {self.target_m:.3f} m)')
            return

        speed = self.speed_mps
        if remaining < SLOWDOWN_MARGIN_M:
            scale = remaining / SLOWDOWN_MARGIN_M
            speed = max(MIN_SPEED_MPS, self.speed_mps * scale)

        self._publish(self.direction * speed)

    def _stop(self, reason: str) -> None:
        self.finished = True
        # Several zeros, since a single dropped message would leave the robot
        # driving. The firmware's 500ms stale-command fail-safe is the backstop,
        # not the primary stop.
        for _ in range(5):
            self._publish(0.0)
        self.get_logger().info(reason)
        self.get_logger().info(
            f'>>> NOW TAPE-MEASURE the actual distance. '
            f'correction = measured / {self.target_m:.3f}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Drive a straight line for a set distance, '
                    'closed-loop on wheel odometry.')
    parser.add_argument('distance', type=float,
                        help='metres; negative drives in reverse')
    parser.add_argument('--speed', type=float, default=0.15,
                        help='m/s, default 0.15')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='seconds before giving up, default 60')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = DriveDistance(args.distance, args.speed, args.timeout)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        # Let the trailing zero-velocity messages actually go out.
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node._stop('interrupted')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
