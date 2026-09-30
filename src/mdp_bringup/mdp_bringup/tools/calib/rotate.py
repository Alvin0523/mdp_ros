"""Rotate the robot heading by a target angle (90 to 360 degrees), closed-loop on IMU / EKF yaw.

WHY THIS EXISTS: Ackermann steering vehicles cannot rotate on the spot (no zero-radius turn).
To achieve a heading change, the car drives an arc at steering lock. This node tracks
accumulated yaw from /odometry/filtered (or /imu/data) with angle unwrapping (allowing 180, 270,
and 360 degree rotations without discontinuity), decelerates smoothly as the target angle
approaches, and halts with zero steering offset when the target heading is reached.

Assessment requirement: CCDS MDP Module A.4 (Accurate Rotation 90 - 360 deg taking Ackermann
turning radius into account).

GYRO DRIFT: first the car stands still for 3 s and the mean IMU turn rate is
taken - the gyro's bias, which the EKF adds up into heading error over a run.

THE LOG: at the end it asks what a protractor says it turned (in sim it takes
Gazebo's true heading) and adds a row to calibration_log.csv (tools/calib/log.py).

Usage:
    pixi run calib rotate 90
    pixi run calib rotate 180 --direction right
    pixi run calib rotate 360 --speed 0.12
"""

import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tools.calib import log

CMD_TOPIC = '/cmd_vel'
ODOM_TOPIC = '/odometry/filtered'
PUBLISH_HZ = 20.0

WHEELBASE_M = planner_params.car_from_urdf()['wheelbase']   # the URDF
DEFAULT_STEER_DEG = 28.0  # Safely inside chassis lock (left +43 deg, right -32.5 deg, re-measured 2026-09-18)
SLOWDOWN_MARGIN_DEG = 15.0
MIN_SPEED_MPS = 0.05
PROGRESS_LOG_HZ = 2.0
STILL_S = 3.0           # standing still first, for the gyro drift
# Progress falling this far past zero, and staying there, means the tracked
# angle is growing the wrong way (e.g. the EKF fell back to raw wheel
# odometry with a sign convention opposite this node's) - the arc will only
# circle wider, not close in on the target, so stop instead of burning the
# full timeout.
SIGN_INVERSION_DEG = 10.0
SIGN_INVERSION_TICKS = int(0.5 * PUBLISH_HZ)


def yaw_from_quaternion(q) -> float:
    """2D yaw in radians from a geometry_msgs/Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RotateAngle(Node):
    def __init__(self, target_deg: float, direction: str, speed_mps: float,
                 steer_deg: float, timeout_s: float, topic: str = ODOM_TOPIC):
        super().__init__('calib_rotate')

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
        self.last_log_time = self.start_time
        self.inversion_ticks = 0
        self.gyro = []                   # IMU yaw rates while standing still
        self.moving = False
        self.truth = log.Truth(self)
        self.truth_prev = None
        self.truth_turned = 0.0          # rad, unwrapped, Gazebo (sim)
        self.create_subscription(Imu, '/imu/data', self._on_imu, 50)

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

    def _on_imu(self, msg: Imu) -> None:
        if not self.moving:
            self.gyro.append(msg.angular_velocity.z)

    def gyro_drift_deg_s(self):
        return math.degrees(sum(self.gyro) / len(self.gyro)) if self.gyro else None

    def _track_truth(self) -> None:
        if self.truth.pose is None:
            return
        yaw = self.truth.pose[2]
        if self.truth_prev is not None and self.moving:
            self.truth_turned += math.atan2(math.sin(yaw - self.truth_prev), math.cos(yaw - self.truth_prev))
        self.truth_prev = yaw

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

        self._track_truth()
        if not self.moving:
            # Stand still first: the gyro's drift, then start counting from here.
            self._publish(0.0, 0.0)
            if elapsed < STILL_S:
                return
            self.moving = True
            self.accumulated_rad = 0.0
            drift = self.gyro_drift_deg_s()
            self.get_logger().info('GYRO      no IMU on /imu/data' if drift is None else
                                   f'GYRO      drift standing still {drift:+.3f} deg/s '
                                   f'({drift * 60:+.1f} deg per minute)')

        progress = self.progress_rad()
        remaining_rad = self.target_rad - progress
        remaining_deg = math.degrees(remaining_rad)
        progress_deg = math.degrees(progress)

        if remaining_deg <= 0.0:
            final_deg = progress_deg
            self._stop(f'DONE - completed rotation of {final_deg:.1f} deg '
                       f'(target {self.target_deg:.1f} deg, error {final_deg - self.target_deg:+.1f} deg)')
            return

        if progress_deg < -SIGN_INVERSION_DEG:
            self.inversion_ticks += 1
            if self.inversion_ticks >= SIGN_INVERSION_TICKS:
                self._stop(
                    f'ABORT - tracked angle is going backwards '
                    f'({progress_deg:.1f} deg after {elapsed:.1f}s instead of '
                    f'increasing toward {self.target_deg:.1f} deg). Likely a sign '
                    f'mismatch between this node\'s direction convention and '
                    f'{self.odom_topic}\'s yaw, or the EKF fusing bad/absent IMU '
                    f'data and falling back to wheel odometry.')
                return
        else:
            self.inversion_ticks = 0

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds / 1e9 >= 1.0 / PROGRESS_LOG_HZ:
            self.last_log_time = now
            raw_yaw = self.prev_yaw if self.prev_yaw is not None else 0.0
            self.get_logger().info(
                f'Progress: {progress_deg:+.1f} deg / {self.target_deg:.1f} deg '
                f'(raw yaw: {raw_yaw:+.2f} rad, remaining: {remaining_deg:.1f} deg)')

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

    def write_log(self) -> None:
        """The run's row in calibration_log.csv (protractor asked for on the real car)."""
        where = log.where(self)
        car = math.degrees(self.progress_rad())
        if where == 'sim' and self.truth_prev is not None:
            true = math.degrees(self.truth_turned if self.is_left else -self.truth_turned)
        else:
            true = log.ask('PROTRACTOR how many degrees did it really turn')
        drift = self.gyro_drift_deg_s()
        notes = f'{"left" if self.is_left else "right"}, steer {math.degrees(self.steer_rad):.0f} deg'
        if drift is not None:
            notes += f'; gyro drift {drift:+.3f} deg/s'
        log.write(where, 'rotate', f'{self.target_deg:.0f} deg', self.speed_mps, car, true, 'deg', notes)

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
        # Let the zeros go out and the car settle, then log.
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
            node._track_truth()
        if node.moving:
            node.write_log()
    except KeyboardInterrupt:
        node._stop('Interrupted by user')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
