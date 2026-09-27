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
"""
import argparse
import math

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_msgs.msg import TFMessage

from mdp_algorithm.control.pure_pursuit_follower import yaw_from_quaternion
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils.run import run, wall_timer

SETTLE_DEG = 30.0      # left out of the circle fit: the steering is still swinging over


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
        self.ekf.add(p.position.x, p.position.y, self.turned)

    def on_truth(self, msg: TFMessage):
        if msg.transforms and self.prev_yaw is not None and not self.done:
            t = msg.transforms[0].transform.translation    # the car (base_footprint)
            self.truth.add(t.x, t.y, self.turned)

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def tick(self):
        if self.prev_yaw is None or self.done:
            return
        if abs(self.turned) < self.angle:
            self.send(self.speed, self.w)
            return
        self.send(0.0, 0.0)
        self.done = True
        lines = [f"STOPPED   after {abs(self.turned):.0f} deg - mark the floor again",
                 self.ekf.report('EKF estimate', self.angle), self.truth.report('Gazebo (true)', self.angle)]
        key = f"minimum_turning_radius_{self.side}"
        lines.append(f"  -> config/navigation.yaml robot.{key}: <tape diameter / 2, in m>")
        for line in filter(None, lines):
            self.get_logger().info(line)
        raise SystemExit

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
