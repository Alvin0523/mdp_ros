#!/usr/bin/env python3
"""Rotate the robot heading by a target angle (90 to 360 degrees), closed-loop on IMU / EKF yaw.

WHY THIS EXISTS: Ackermann steering vehicles cannot rotate on the spot (no zero-radius turn).
To achieve a heading change, the car drives an arc at steering lock. This node tracks
accumulated yaw from /odometry/filtered (or /imu/data) with angle unwrapping (allowing 180, 270,
and 360 degree rotations without discontinuity), decelerates smoothly as the target angle
approaches, and halts with zero steering offset when the target heading is reached.

Assessment requirement: CCDS MDP Module A.4 (Accurate Rotation 90 - 360 deg taking Ackermann
turning radius into account).

Usage:
    ros2 run mdp_bringup rotate_angle.py 90
    ros2 run mdp_bringup rotate_angle.py 180 --direction right
    ros2 run mdp_bringup rotate_angle.py 360 --speed 0.12
    pixi run rotate 90
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
ODOM_TOPIC = '/odometry/filtered'
PUBLISH_HZ = 20.0

WHEELBASE_M = 0.1433  # WHEELTEC C30D Ackermann wheelbase
DEFAULT_STEER_DEG = 28.0  # Safely inside chassis lock (left +35 deg, right -29.5 deg)
SLOWDOWN_MARGIN_DEG = 15.0
MIN_SPEED_MPS = 0.05


def yaw_from_quaternion(q) -> float:
    """2D yaw in radians from a geometry_msgs/Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RotateAngle(Node):
    def __init__(self, target_deg: float, direction: str, speed_mps: float,
                 steer_deg: float, timeout_s: float, topic: str = ODOM_TOPIC):
        super().__init__('rotate_angle')

        self.target_deg = abs(target_deg)
        self.target_rad = math.radians(self.target_deg)
        self.is_left = (direction.lower() == 'left')
        self.speed_mps = abs(speed_mps)
        self.steer_rad = math.radians(steer_deg)
        self.timeout_s = timeout_s
        self.odom_topic = topic

        self.prev_yaw = None
        self.start_yaw = None
        self.accumulated_rad = 0.0
        self.finished = False
        self.start_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, 10)

        self.create_subscription(
            Odometry, self.odom_topic, self._on_odom,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        turn_str = "LEFT (CCW)" if self.is_left else "RIGHT (CW)"
        self.get_logger().info(
            f'Target rotation: {self.target_deg:.1f} deg {turn_str} '
            f'at speed {self.speed_mps:.2f} m/s, steering {steer_deg:.1f} deg, '
            f'tracking {self.odom_topic}')

    def _on_odom(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if self.prev_yaw is None:
            self.prev_yaw = yaw
            self.start_yaw = yaw
            return

        # Shortest-angle difference across [-pi, pi] boundary for continuous unwrapping
        diff = math.atan2(math.sin(yaw - self.prev_yaw), math.cos(yaw - self.prev_yaw))
        self.accumulated_rad += diff
        self.prev_yaw = yaw

    def _publish(self, vx: float, wz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx
        msg.twist.angular.z = wz
        self.cmd_pub.publish(msg)

    def _tick(self) -> None:
        if self.finished:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed > self.timeout_s:
            rot_deg = math.degrees(self.progress_rad())
            self._stop(f'TIMEOUT after {elapsed:.1f}s - '
                       f'rotated {rot_deg:.1f} deg of {self.target_deg:.1f} deg')
            return

        if self.prev_yaw is None:
            self._publish(0.0, 0.0)
            if elapsed > 5.0:
                self._stop(f'No odometry on {self.odom_topic} after 5s - is bringup running?')
            return

        progress = self.progress_rad()
        remaining_rad = self.target_rad - progress
        remaining_deg = math.degrees(remaining_rad)

        if remaining_deg <= 0.0:
            final_deg = math.degrees(progress)
            self._stop(f'DONE - completed rotation of {final_deg:.1f} deg '
                       f'(target {self.target_deg:.1f} deg, error {final_deg - self.target_deg:+.1f} deg)')
            return

        # Slowdown profiling near completion
        speed = self.speed_mps
        if remaining_deg < SLOWDOWN_MARGIN_DEG:
            scale = max(0.0, remaining_deg / SLOWDOWN_MARGIN_DEG)
            speed = max(MIN_SPEED_MPS, self.speed_mps * scale)

        # Bicycle model: omega_z = vx * tan(steer) / L
        steer_sign = 1.0 if self.is_left else -1.0
        applied_steer = steer_sign * self.steer_rad
        omega_z = speed * math.tan(applied_steer) / WHEELBASE_M

        self._publish(speed, omega_z)

    def progress_rad(self) -> float:
        """Absolute rotation in the intended direction (radians)."""
        return self.accumulated_rad if self.is_left else -self.accumulated_rad

    def _stop(self, reason: str) -> None:
        self.finished = True
        # Send zero velocity & center steering multiple times to flush pipeline
        for _ in range(5):
            self._publish(0.0, 0.0)
        self.get_logger().info(reason)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Rotate heading by a set angle (90-360 deg) along an Ackermann arc.')
    parser.add_argument('angle', type=float,
                        help='degrees to rotate (e.g. 90, 180, 270, 360); negative implies right')
    parser.add_argument('--direction', type=str, default=None, choices=['left', 'right'],
                        help='turn direction: left (CCW) or right (CW). Default: left (or right if angle < 0)')
    parser.add_argument('--speed', type=float, default=0.15,
                        help='linear forward speed in m/s, default 0.15')
    parser.add_argument('--steer', type=float, default=DEFAULT_STEER_DEG,
                        help=f'steering angle in degrees, default {DEFAULT_STEER_DEG}')
    parser.add_argument('--timeout', type=float, default=45.0,
                        help='seconds before giving up, default 45')
    parser.add_argument('--topic', type=str, default=ODOM_TOPIC,
                        help=f'odometry topic (default: {ODOM_TOPIC})')
    args, ros_args = parser.parse_known_args()

    # Determine direction from sign or explicit flag
    if args.direction is not None:
        direction = args.direction
    else:
        direction = 'right' if args.angle < 0.0 else 'left'

    rclpy.init(args=ros_args)
    node = RotateAngle(
        target_deg=abs(args.angle),
        direction=direction,
        speed_mps=args.speed,
        steer_deg=args.steer,
        timeout_s=args.timeout,
        topic=args.topic
    )
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        node._stop('Interrupted by user')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
