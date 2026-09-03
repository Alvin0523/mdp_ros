#!/usr/bin/env python3
"""
Pure Pursuit Path Follower for Ackermann Steering Controller.
Calculates steering angle (angular.z) and velocity (linear.x) to track path waypoints.

Split into two pieces:
- PurePursuitController: plain, ROS-free pursuit math + path/pose state.
  Reusable by any caller that already has its own localization
  subscription and /cmd_vel publisher (e.g. task1_runner.py, so it doesn't
  need a second competing node publishing to /cmd_vel).
- PurePursuitFollower(Node): a standalone ROS2 node wrapping the
  controller for direct use (subscribes /odometry/filtered, runs its own
  control loop, publishes /cmd_vel itself) - kept for manual/standalone
  testing via `ros2 run mdp_algorithm pure_pursuit_follower`, not launched
  by real.launch.py/sim.launch.py by default.

COMPLETED 2026-09-03 (was previously a skeleton): calculate_pure_pursuit()'s
steering-law math was already correct, but nothing actually called it - no
control loop, no localization subscription (joint_states_cb was a no-op,
current_pose was never updated from anything), no lookahead-point search
along the path, and no forward/reverse conversion from steering angle to
the angular.z (yaw rate) /cmd_vel actually expects. All of that is added
here; see control_loop()/find_lookahead_point()/steering-to-omega in
compute_cmd() below.
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

Pose = Tuple[float, float, float]


def yaw_from_quaternion(q) -> float:
    """2D yaw from a geometry_msgs/Quaternion (only the z-rotation component
    matters for a ground vehicle)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PurePursuitController:
    """ROS-free pure-pursuit path follower: feed it a path and a stream of
    pose updates, pull a (linear_x, angular_z) /cmd_vel command each tick."""

    def __init__(self, wheelbase: float = 0.1433, lookahead_dist: float = 0.25,
                 max_steering_angle: float = 0.5672, target_speed: float = 0.5,
                 goal_tolerance: float = 0.05):
        self.wheelbase = wheelbase
        self.lookahead_dist = lookahead_dist
        self.max_steering_angle = max_steering_angle
        self.target_speed = target_speed
        self.goal_tolerance = goal_tolerance

        self.path: List[Pose] = []
        self.current_pose: Pose = (0.0, 0.0, 0.0)
        self.active = False
        self._search_idx = 0   # monotonic - never re-scans behind where we already passed

    def set_path(self, path_waypoints: List[Pose]) -> None:
        self.path = list(path_waypoints)
        self._search_idx = 0
        self.active = bool(self.path)

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.current_pose = (x, y, yaw)

    def is_done(self) -> bool:
        return not self.active

    def find_lookahead_point(self) -> Optional[Tuple[float, float]]:
        """First path point at least lookahead_dist ahead of the current
        pose, starting the search from the last point found (monotonic, so
        the follower can't get stuck re-targeting a point it already
        passed). Falls back to the final waypoint once no point further out
        remains - lets calculate_pure_pursuit() home in on the exact goal
        rather than overshooting past it."""
        if not self.path:
            return None

        for i in range(self._search_idx, len(self.path)):
            px, py, _ = self.path[i]
            if math.hypot(px - self.current_pose[0], py - self.current_pose[1]) >= self.lookahead_dist:
                self._search_idx = i
                return px, py

        self._search_idx = len(self.path) - 1
        return self.path[-1][0], self.path[-1][1]

    def calculate_pure_pursuit(self, target_point: Tuple[float, float]) -> float:
        dx = target_point[0] - self.current_pose[0]
        dy = target_point[1] - self.current_pose[1]

        yaw = self.current_pose[2]
        local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        if local_x <= 0:
            return 0.0

        curvature = (2.0 * local_y) / (self.lookahead_dist ** 2)
        steering_angle = math.atan(self.wheelbase * curvature)
        return max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))

    def compute_cmd(self) -> Optional[Tuple[float, float]]:
        """One control-loop tick. Returns (linear_x, angular_z) to publish
        to /cmd_vel, or None if there's nothing to do right now (inactive/
        no path) - caller should hold last command or stop in that case.
        Sets is_done()==True once the final waypoint is reached."""
        if not self.active or not self.path:
            return None

        goal_x, goal_y, _ = self.path[-1]
        if math.hypot(goal_x - self.current_pose[0], goal_y - self.current_pose[1]) <= self.goal_tolerance:
            self.active = False
            return 0.0, 0.0

        target = self.find_lookahead_point()
        if target is None:
            self.active = False
            return 0.0, 0.0

        steering_angle = self.calculate_pure_pursuit(target)
        # ackermann_steering_controller's /cmd_vel takes body yaw rate
        # (angular.z = omega), not the steering angle itself - convert via
        # the standard bicycle-model relation omega = v*tan(delta)/L.
        angular_z = self.target_speed * math.tan(steering_angle) / self.wheelbase
        return self.target_speed, angular_z


class PurePursuitFollower(Node):
    """Standalone node wrapper - see module docstring. Not part of the
    default launch graph; task1_runner.py uses PurePursuitController
    directly instead so there's only one /cmd_vel publisher during Task 1."""

    def __init__(self):
        super().__init__('pure_pursuit_follower')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        self.controller = PurePursuitController()

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)
        self.create_timer(0.05, self.control_loop)  # 20Hz, matches task1_runner's rate

    def odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.controller.update_pose(p.x, p.y, yaw)

    def set_path(self, path_waypoints: List[Pose]) -> None:
        self.controller.set_path(path_waypoints)

    def control_loop(self) -> None:
        cmd = self.controller.compute_cmd()
        if cmd is None:
            return
        self.publish_cmd(*cmd)

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop(self) -> None:
        self.controller.active = False
        self.publish_cmd(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
