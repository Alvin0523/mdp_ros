"""Drive a straight line for a set distance, closed-loop on wheel odometry.

WHY THIS EXISTS: ROS has no "move 2 metres" command. /cmd_vel is a velocity,
so publishing it for a fixed duration is open-loop in time and depends on
battery voltage, load and how fast the PID settles. This node instead watches
odometry and stops when the distance is actually reported as covered.

PRIMARY USE - calibrating the distance scale. Wheel odometry is computed from
encoder ticks and wheel radius:

    metres_per_tick = 2*pi*wheel_radius / ticks_per_wheel_rev
                    = 2*pi*0.0331 / 1560  ~= 0.133 mm

wheel_radius is the URDF's (0.0331 m: set 2026-09-27 from this test - 4 x 2.0 m,
the car went 1.018x what the old 65 mm tyre predicted); ticks_per_wheel_rev is
WHEELTEC's. To re-check, run this with a target, then TAPE-MEASURE what the
robot actually travelled:

    correction = measured_distance / requested_distance

Anything other than ~1.0 means the constants are wrong by that factor. Note
odometry only depends on the PRODUCT 2*pi*r/ticks_per_rev, so this one test
gives the working correction even without knowing which of the two is off.
Wheel radius lives in the URDF (wheel_radius) and ticks-per-rev in
mdp_stm32/src/motor.c (MOTOR_TICKS_PER_REV).

AT SPEED: --speed 0.9 checks the distance still holds at task 2's speed (wheel
slip) and prints the ROLL-PAST: how far the car kept going after the stop
command. A short run may not reach the speed asked (soft start, --accel); the
log records the peak speed actually reached.

THE LOG: at the end it asks for the tape distance and the sideways drift (in
sim it takes both from Gazebo's true pose) and adds a row to
calibration_log.csv (tools/calib/log.py).

Deliberately subscribes to the CONTROLLER's raw odometry rather than
/odometry/filtered: the EKF fuses IMU yaw rate on top, and for calibrating a
distance scale we want the quantity that is a direct function of the encoder
ticks and wheel radius, with nothing else mixed in.

SECONDARY USE - steering-center calibration. servo_set_angle(0) is supposed
to point the wheels dead straight (SERVO_PULSE_CENTER_US in
mdp_stm32/include/servo.h, 1490us, measured by hand-pushing the car with the
motors off). Under real driving load that static measurement can be off,
which shows up here as the car curving during an otherwise-straight run.
--steer-deg commands a small constant steering bias (through the same
bicycle-model path ackermann_steering_controller and `calib rotate` use, not
a raw PWM override) so you can binary-search for the angle that actually
cancels the curve against a real straight-edge on the floor, then convert
that angle to a pulse offset and fix SERVO_PULSE_CENTER_US itself instead of
carrying a trim flag forever.

Usage:
    pixi run calib straight 2.0
    pixi run calib straight 2.0 --speed 0.1
    pixi run calib straight 2.0 --speed 0.9 --accel 0.5   # task 2 speed
    pixi run calib straight -1.0          # reverse
    pixi run calib straight 2.0 --steer-deg -1.5   # trim test

Requires the bare car to be running (pixi run pi / pi-solo / sim, task:=0) and the
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
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tools.calib import log

CMD_TOPIC = '/cmd_vel'
ODOM_TOPIC = '/ackermann_steering_controller/odometry'
PUBLISH_HZ = 20.0

WHEELBASE_M = planner_params.car_from_urdf()['wheelbase']   # the URDF

# After the stop command, keep watching this long for the roll-past.
SETTLE_S = 1.5

# Stop ramping down this far out so the car coasts onto the target rather than
# overshooting it - the drivetrain cannot stop instantly.
SLOWDOWN_MARGIN_M = 0.10
MIN_SPEED_MPS = 0.05

# Soft start: speed rises at this rate (m/s^2) instead of stepping instantly to
# --speed, which made the rear wheels slip at the start. 0.15 m/s -> ~1 s ramp.
DEFAULT_ACCEL_MPS2 = 0.15


class DriveDistance(Node):
    def __init__(self, target_m: float, speed_mps: float, timeout_s: float,
                 steer_deg: float = 0.0, accel_mps2: float = DEFAULT_ACCEL_MPS2):
        super().__init__('calib_straight')

        self.target_m = abs(target_m)
        self.direction = 1.0 if target_m >= 0.0 else -1.0
        self.speed_mps = abs(speed_mps)
        self.timeout_s = timeout_s
        self.steer_rad = math.radians(steer_deg)
        self.accel_mps2 = abs(accel_mps2)
        self.ramp_speed = 0.0

        self.start_xy = None
        self.travelled_m = 0.0
        self.finished = False
        self.stop_travelled_m = None     # odometry distance when the stop was sent
        self.peak_speed = 0.0
        self.truth = log.Truth(self)
        self.truth_start = None
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
            f'at {self.speed_mps:.2f} m/s, steer trim {math.degrees(self.steer_rad):+.1f} deg, '
            f'odom from {ODOM_TOPIC}')

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.peak_speed = max(self.peak_speed, abs(msg.twist.twist.linear.x))
        if self.start_xy is None:
            self.start_xy = (x, y)
            self.truth_start = self.truth.pose
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
        # Bicycle model, same as `calib rotate` - vx already carries the
        # direction sign, so a trim commanded for forward driving flips the
        # correct way in reverse too.
        msg.twist.angular.z = (
            vx * math.tan(self.steer_rad) / WHEELBASE_M if self.steer_rad else 0.0)
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

        self.ramp_speed = min(self.speed_mps,
                              self.ramp_speed + self.accel_mps2 / PUBLISH_HZ)
        speed = self.ramp_speed
        if remaining < SLOWDOWN_MARGIN_M:
            scale = remaining / SLOWDOWN_MARGIN_M
            speed = min(speed, max(MIN_SPEED_MPS, self.speed_mps * scale))

        self._publish(self.direction * speed)

    def _stop(self, reason: str) -> None:
        self.finished = True
        self.stop_travelled_m = self.travelled_m
        # Several zeros, since a single dropped message would leave the robot
        # driving. The firmware's 500ms stale-command fail-safe is the backstop,
        # not the primary stop.
        for _ in range(5):
            self._publish(0.0)
        self.get_logger().info(reason)

    def true_move(self):
        """Sim: Gazebo's (along, sideways-left) movement since the start, metres."""
        if self.truth_start is None or self.truth.pose is None:
            return None
        x0, y0, yaw0 = self.truth_start
        dx, dy = self.truth.pose[0] - x0, self.truth.pose[1] - y0
        return (dx * math.cos(yaw0) + dy * math.sin(yaw0),
                -dx * math.sin(yaw0) + dy * math.cos(yaw0))

    def write_log(self):
        """The run's row in calibration_log.csv (tape asked for on the real car)."""
        where = log.where(self)
        car_cm = self.travelled_m * 100.0
        roll_cm = (self.travelled_m - self.stop_travelled_m) * 100.0
        print(f'CAR       odometry {car_cm:.1f} cm (roll-past after the stop {roll_cm:.1f} cm), '
              f'peak {self.peak_speed:.2f} m/s')
        move = self.true_move() if where == 'sim' else None
        if move is not None:
            true_cm, side_cm = abs(move[0]) * 100.0, move[1] * 100.0
        else:
            true_cm = log.ask('TAPE      distance driven (cm)')
            side_cm = log.ask('TAPE      sideways drift at the end (cm, + = left)')
        notes = f'roll-past {roll_cm:.1f} cm'
        if side_cm is not None:
            notes += f'; sideways {side_cm:+.1f} cm'
        if self.steer_rad:
            notes += f'; steer trim {math.degrees(self.steer_rad):+.1f} deg'
        if true_cm:
            notes += f'; correction {true_cm / car_cm:.3f}'
        log.write(where, 'straight', f'{self.direction * self.target_m:.2f} m', self.peak_speed,
                  car_cm, true_cm, 'cm', notes)


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
    parser.add_argument('--accel', type=float, default=DEFAULT_ACCEL_MPS2,
                        help='start-up acceleration in m/s^2 (default %(default)s)')
    parser.add_argument('--steer-deg', type=float, default=0.0,
                        help='constant steering trim in degrees for '
                             'straight-line calibration, default 0 '
                             '(positive = left, per REP-103)')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = DriveDistance(args.distance, args.speed, args.timeout, args.steer_deg,
                         args.accel)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        # Let the zeros go out, and watch the roll-past until the car is still.
        end = node.get_clock().now().nanoseconds / 1e9 + SETTLE_S
        while rclpy.ok() and node.get_clock().now().nanoseconds / 1e9 < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.start_xy is not None:
            node.write_log()
    except KeyboardInterrupt:
        node._stop('interrupted')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
