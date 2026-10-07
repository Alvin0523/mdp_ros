"""`calib turn`: the real turning circle at full lock, per side - Nav2's
minimum_turning_radius_left / _right in config/navigation.yaml.

    pixi run calib turn left
    pixi run calib turn right
    pixi run calib turn left --angle 360 --speed 0.2

HOW TO MEASURE: mark the floor under the middle of the rear axle, run it, mark
again where it stops. After half a circle (the default, 180 deg) the two marks
are one DIAMETER apart: radius = the tape distance / 2.

Drives at full lock (the URDF's steering limits) at the planner's speed, 0.2
m/s by default - the circle grows with speed, so measure at the speed the car
will drive. Stops on the EKF heading. Prints the same radius from the EKF track
(a circle fit, first 30 deg left out while the steering swings over) and, in
sim, from Gazebo's true pose - compare with the tape.

STEERING DELAY: the time from the full-lock command until the car turns at
90 % of its steady rate - the servo swinging over plus the motor speeding up.

THE LOG: at the end it asks for the tape distance (in sim it takes Gazebo's
true circle) and adds a row to calibration_log.csv (tools/calib/log.py).
"""
import argparse
import math

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_msgs.msg import TFMessage

from mdp_algorithm.control.path_follower import yaw_from_quaternion
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tools.calib import log
from mdp_bringup.utils.run import run, wall_timer

SETTLE_DEG = 30.0      # left out of the circle fit: the steering is still swinging over
SLOW_DEG = 30.0        # slow down over the last this-many degrees (still full lock), as
                       # calib rotate: at 0.4 m/s it stopped late and rolled on (real car,
                       # 2026-10-01)
MIN_SPEED = 0.08       # m/s, the slowest it creeps to the stop
STOP_HOLD_S = 1.0      # zeros this long after stopping (and the car settles) before reporting


def fit_radius(points) -> float:
    """Least-squares circle through (x, y) points (Kasa fit), metres."""
    p = np.asarray(points, dtype=float)
    if len(p) < 5:
        return float('nan')
    a = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    b = (p ** 2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    return float(math.sqrt(c + cx ** 2 + cy ** 2))


class Track:
    """One pose source's track: start point, points after SETTLE_DEG, last point."""

    def __init__(self):
        self.start = None
        self.last = None
        self.points = []

    def add(self, x, y, turned_deg):
        if self.start is None:
            self.start = (x, y)
        self.last = (x, y)
        if abs(turned_deg) >= SETTLE_DEG:
            self.points.append((x, y))

    def radius(self, angle) -> float:
        """metres: half the start->stop distance after 180 deg, else the circle fit."""
        if self.start is None:
            return float('nan')
        return math.dist(self.start, self.last) / 2.0 if abs(angle - 180.0) < 1.0 else fit_radius(self.points)

    def report(self, name, angle):
        if self.start is None:
            return None
        chord = math.dist(self.start, self.last)
        line = f"  {name:<14} fitted radius {fit_radius(self.points) * 100:5.1f} cm"
        if abs(angle - 180.0) < 1.0:
            line += f"   start->stop {chord * 100:5.1f} cm = diameter -> radius {chord * 50:5.1f} cm"
        return line


class Turn(Node):
    def __init__(self, side, angle, speed):
        super().__init__('calib_turn')
        car = planner_params.car_from_urdf()
        lock = car['steering_limit_left'] if side == 'left' else -car['steering_limit_right']
        self.side, self.angle, self.speed = side, angle, speed
        self.w = speed * math.tan(lock) / car['wheelbase']      # bicycle model, rad/s
        self.lock_deg = math.degrees(lock)
        self.yaw0 = self.prev_yaw = None
        self.turned = 0.0            # deg, unwrapped, since the start
        self.ekf, self.truth = Track(), Track()
        self.done = False
        self.t0 = None               # first full-lock command, s
        self.stop_time = None        # when it reached the angle and started stopping
        self.turned_at_stop = 0.0
        self.rates = []              # (s since t0, |yaw rate| rad/s) from the EKF
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 10)
        self.create_subscription(TFMessage, '/sim/ground_truth', self.on_truth, 10)   # sim only
        wall_timer(self, 0.05, self.tick)
        self.get_logger().info(f"TURN      {side} at full lock ({self.lock_deg:+.1f} deg), {speed:.2f} m/s, "
                               f"{angle:.0f} deg - mark the floor under the middle of the rear axle")

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        yaw = yaw_from_quaternion(p.orientation)
        if self.prev_yaw is not None:
            self.turned += math.degrees(math.atan2(math.sin(yaw - self.prev_yaw), math.cos(yaw - self.prev_yaw)))
        self.prev_yaw = yaw
        if self.t0 is not None and self.stop_time is None and self.angle - abs(self.turned) >= SLOW_DEG:
            self.rates.append((self.now() - self.t0, abs(msg.twist.twist.angular.z)))
        self.ekf.add(p.position.x, p.position.y, self.turned)

    def on_truth(self, msg: TFMessage):
        if msg.transforms and self.prev_yaw is not None and not self.done:
            t = msg.transforms[0].transform.translation    # the car (base_footprint)
            self.truth.add(t.x, t.y, self.turned)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def steering_delay(self):
        """s from the command to 90 % of the steady turn rate (the rate after SETTLE_DEG)."""
        if len(self.rates) < 10:
            return None
        steady = float(np.median([r for _, r in self.rates[len(self.rates) // 2:]]))
        return next((t for t, r in self.rates if r >= 0.9 * steady), None)

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def tick(self):
        if self.prev_yaw is None or self.done:
            return
        left = self.angle - abs(self.turned)
        if self.stop_time is None and left > 0.0:
            if self.t0 is None:
                self.t0 = self.now()
            v = self.speed if left >= SLOW_DEG else max(MIN_SPEED, self.speed * left / SLOW_DEG)
            self.send(v, self.w * v / self.speed)      # same full-lock circle, slower
            return
        # Stopped: keep sending zeros (one lost stop message left it rolling) and
        # let the car settle before reading how far it really turned.
        self.send(0.0, 0.0)
        if self.stop_time is None:
            self.stop_time = self.now()
            self.turned_at_stop = abs(self.turned)
        if self.now() - self.stop_time < STOP_HOLD_S:
            return
        self.done = True
        lines = [f"STOPPED   after {abs(self.turned):.0f} deg ({abs(self.turned) - self.turned_at_stop:+.0f} deg "
                 f"rolled on after the stop) - mark the floor again",
                 self.ekf.report('EKF estimate', self.angle), self.truth.report('Gazebo (true)', self.angle)]
        delay = self.steering_delay()
        if delay is not None:
            lines.append(f"  steering delay {delay:.2f} s (command -> 90 % of the full turn rate)")
        for line in filter(None, lines):
            self.get_logger().info(line)
        self.send(0.0, 0.0)
        self.write_log()     # the car is stopped; the tape question can wait for an answer
        raise SystemExit

    def write_log(self):
        """The run's row in calibration_log.csv (tape asked for on the real car)."""
        where = log.where(self)
        car = self.ekf.radius(self.angle) * 100.0
        if where == 'sim' and self.truth.start is not None:
            true = self.truth.radius(self.angle) * 100.0
        elif abs(self.angle - 180.0) < 1.0:
            tape = log.ask('TAPE      start mark -> stop mark (cm)')
            true = None if tape is None else tape / 2.0
        else:
            true = log.ask('TAPE      radius of the circle the rear axle drew (cm)')
        delay = self.steering_delay()
        notes = f'{self.side}, full lock {self.lock_deg:+.1f} deg, {self.angle:.0f} deg turned'
        if delay is not None:
            notes += f'; steering delay {delay:.2f} s'
        log.write(where, f'turn {self.side}', f'{self.angle:.0f} deg', self.speed, car, true, 'cm radius', notes)
        print(f'          set by hand: config/navigation.yaml robot.minimum_turning_radius_{self.side} '
              f'(metres, the TRUE radius)')

    def destroy_node(self):
        try:
            self.send(0.0, 0.0)   # never leave the car driving (Ctrl+C)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    ap = argparse.ArgumentParser(description='Turning circle at full lock: calib turn left|right')
    ap.add_argument('side', choices=['left', 'right'])
    ap.add_argument('--angle', type=float, default=180.0, help='deg to turn (default 180: stop one diameter away)')
    ap.add_argument('--speed', type=float, default=0.2, help='m/s (default 0.2, the planner speed)')
    a, ros_args = ap.parse_known_args()
    try:
        run(lambda: Turn(a.side, a.angle, a.speed), args=ros_args)
    except SystemExit:
        pass


if __name__ == '__main__':
    main()
