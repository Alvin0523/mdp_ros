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
# (x, y, theta, gear) - gear is +1 forward / -1 reverse (motion_primitives.Gear).
# Matches collision_aware_planner.DensePose; kept as a separate alias here
# since this module is meant to be usable without importing the planner.
DensePose = Tuple[float, float, float, int]


def yaw_from_quaternion(q) -> float:
    """2D yaw from a geometry_msgs/Quaternion (only the z-rotation component
    matters for a ground vehicle)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PurePursuitController:
    """ROS-free pure-pursuit path follower: feed it a path and a stream of
    pose updates, pull a (linear_x, angular_z) /cmd_vel command each tick."""

    # max_steering_angle is a single symmetric bound, so it must use the
    # TIGHTER of the two measured sides or it will command angles the chassis
    # cannot reach in one direction. RE-MEASURED 2026-09-18 (supersedes the
    # 2026-09-11 pair): left +43.0deg (0.7505 rad), right -32.5deg (0.5672
    # rad) - right still binds. Both endpoints (850us left, 2400us right) were
    # re-confirmed as genuine mechanical limits via raw-pulse sweeps past the
    # old calibration ceilings. NOTE: 0.5672/32.5deg coincidentally matches an
    # earlier, since-discredited figure that was never actually measured -
    # see mdp_description's URDF comment for that history; this is an
    # unrelated, freshly-measured value, not a revert.
    # lookahead_dist was 0.25 (25cm) - almost exactly the OLD
    # MIN_TURN_RADIUS_CM (25.3cm; now 22.5cm post re-measurement, see
    # planning_constants.py). Pure pursuit only converges to
    # the path's true curvature when lookahead is meaningfully SMALLER than
    # the turn radius being tracked; at lookahead ~= radius (the case on
    # every corner-avoidance curve the planner produces, since those are
    # planned right up against the vehicle's own turning limit), the
    # controller systematically under-steers relative to the path and cuts
    # the corner toward the outside. Confirmed live (2026-09-17): with
    # 0.25, task1's leg 1 clipped obstacle 1's corner, Gazebo's physics
    # exploded the interpenetration, and the robot flew ~56m off the arena
    # at full commanded speed with no recovery (is_done() never triggers
    # once genuinely lost - see the note in find_lookahead_point()).
    # Reduced further (2026-09-17, second pass): 0.15 alone still wasn't
    # enough - live-observed the car driving straight at a far point across
    # a tight obstacle-avoidance curve, then snap-correcting, instead of a
    # smooth arc (find_lookahead_point() searches by straight-line distance,
    # not arc length, so on a tight curve a point on the FAR side can already
    # be >= lookahead away after just 1-2 path points). 0.10 sits well under
    # both MIN_TURN_RADIUS_CM (22.5cm) and the ~15-20cm scale of the actual
    # avoidance curves around a 10cm obstacle cube with inflation margin.
    # target_speed also cut 0.5 -> 0.2: at 20Hz control rate, 0.5 m/s only
    # gives a correction every 2.5cm traveled; 0.2 m/s gives one every 1cm,
    # which matters most exactly on the curves this lookahead cut is meant
    # to track more faithfully.
    def __init__(self, wheelbase: float = 0.1433, lookahead_dist: float = 0.10,
                 max_steering_angle: float = 0.5672, target_speed: float = 0.2,
                 goal_tolerance: float = 0.05):
        self.wheelbase = wheelbase
        self.lookahead_dist = lookahead_dist
        self.max_steering_angle = max_steering_angle
        self.target_speed = target_speed
        self.goal_tolerance = goal_tolerance

        self.path: List[DensePose] = []
        self.current_pose: Pose = (0.0, 0.0, 0.0)
        self.active = False
        self._search_idx = 0   # monotonic - never re-scans behind where we already passed
        self.last_target = None
        self._seg_ends: List[int] = []   # last path index of each same-gear segment
        self._seg = 0                    # segment being driven

    def set_path(self, path_waypoints: List[DensePose]) -> None:
        """path_waypoints: (x, y, theta, gear) tuples - gear +1/-1. A bare
        (x, y, theta) path (no gear) is auto-upgraded to all-forward for
        backward compatibility with any caller that hasn't been updated."""
        self.path = [
            p if len(p) == 4 else (p[0], p[1], p[2], 1)
            for p in path_waypoints
        ]
        self._search_idx = 0
        self.active = bool(self.path)
        # (x, y, gear) the last compute_cmd() steered at - read-only, for
        # logging/visualisation (task1_runner's /run_status).
        self.last_target = None
        # Split at every gear change (cusp). Each segment is driven on its
        # own: the lookahead search never crosses into the next segment, and
        # the next one starts only once this one's end point is reached or
        # passed - see compute_cmd().
        self._seg_ends = [i - 1 for i in range(1, len(self.path))
                          if self.path[i][3] != self.path[i - 1][3]] + [len(self.path) - 1]
        self._seg = 0

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.current_pose = (x, y, yaw)

    def is_done(self) -> bool:
        return not self.active

    def find_lookahead_point(self) -> Optional[Tuple[float, float, int]]:
        """First path point at least lookahead_dist ahead of the current
        pose, starting the search from the last point found (monotonic, so
        the follower can't get stuck re-targeting a point it already
        passed). Falls back to the final waypoint once no point further out
        remains - lets calculate_pure_pursuit() home in on the exact goal
        rather than overshooting past it. Returns (x, y, gear)."""
        if not self.path:
            return None

        seg_end = self._seg_ends[self._seg]
        for i in range(self._search_idx, seg_end + 1):
            px, py, _, gear = self.path[i]
            if math.hypot(px - self.current_pose[0], py - self.current_pose[1]) >= self.lookahead_dist:
                self._search_idx = i
                return px, py, gear

        self._search_idx = seg_end
        px, py, _, gear = self.path[seg_end]
        return px, py, gear

    def _segment_end_reached(self, idx: int, final: bool) -> bool:
        """True once the car is at path[idx] (a segment's last point), or has
        driven past it in that segment's direction while close to it. The
        'passed' half is what stops the car dithering back and forth over a
        point it missed by a few cm sideways. Tighter for the final checkpoint
        than for an intermediate cusp."""
        px, py, ptheta, gear = self.path[idx]
        dx = self.current_pose[0] - px
        dy = self.current_pose[1] - py
        dist = math.hypot(dx, dy)
        if dist <= self.goal_tolerance:
            return True
        along = dx * math.cos(ptheta) + dy * math.sin(ptheta)   # + = ahead of the point along its heading
        passed = along > 0.0 if gear >= 0 else along < 0.0
        return passed and dist <= (2.0 * self.goal_tolerance if final else 1.5 * self.lookahead_dist)

    def calculate_pure_pursuit(self, target_point: Tuple[float, float], gear: int = 1) -> float:
        """gear: +1 forward / -1 reverse (see set_path). BUG FIX
        (2026-09-17): this used to always assume forward motion - any
        REVERSE-gear path segment (HybridAStar genuinely plans these, see
        collision_aware_planner.DensePose's comment) has its lookahead
        target legitimately BEHIND the current heading, which the old
        forward-only `if local_x <= 0: return 0.0` treated as an error and
        drove straight through instead - sending the car forward along a
        curve only valid in reverse."""
        dx = target_point[0] - self.current_pose[0]
        dy = target_point[1] - self.current_pose[1]

        yaw = self.current_pose[2]
        local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        if gear >= 0:
            # FORWARD: target must be ahead of current heading for the
            # curvature formula below to mean anything; refuse to steer
            # rather than compute nonsense for a behind-you point.
            if local_x <= 0:
                return 0.0
            curvature = (2.0 * local_y) / (self.lookahead_dist ** 2)
        else:
            # REVERSE: the vehicle is moving toward -local_x, so a target
            # that's actually ahead (local_x >= 0) is the anomalous case
            # here instead. The curvature is the SAME expression as forward:
            # the arc through the target is one circle whichever way it is
            # driven, and the bicycle model w = v*tan(delta)/L already turns
            # the other way for negative v (compute_cmd() uses signed speed).
            # This used to negate it as well, and the two flips cancelled -
            # reversing steered AWAY from the target (2026-09-24: every
            # reverse segment ended with the target beside the car).
            if local_x >= 0:
                return 0.0
            curvature = (2.0 * local_y) / (self.lookahead_dist ** 2)

        steering_angle = math.atan(self.wheelbase * curvature)
        return max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))

    def compute_cmd(self) -> Optional[Tuple[float, float]]:
        """One control-loop tick. Returns (linear_x, angular_z) to publish
        to /cmd_vel, or None if there's nothing to do right now (inactive/
        no path) - caller should hold last command or stop in that case.
        Sets is_done()==True once the final waypoint is reached.

        BUG FIX (2026-09-17): the goal check used to run BEFORE
        find_lookahead_point(), purely on straight-line distance to the
        final waypoint - so any leg whose planned reverse segment covers
        only the last ~20-30cm (see DensePose's comment - HybridAStar often
        backs into the final checkpoint rather than looping around to face
        it forward) could get marked done from tracking error alone: normal
        lookahead-pursuit corner-cutting on the FORWARD portion is enough to
        land within goal_tolerance (5cm) of the final point before the
        reverse maneuver ever starts, silently skipping it - looks exactly
        like "the car never reverses" even though the planner asked for it
        (confirmed the planner+follower math itself is correct via a
        perfect-tracking replay - this was the only place real-world
        tracking noise could still substitute forward corner-cutting for a
        planned reverse). Fixed by tying "done" to having actually
        consumed the whole path (find_lookahead_point() exhausted, i.e.
        it's returning the final waypoint itself because nothing farther
        out remains) rather than just being spatially close to the end."""
        if not self.active or not self.path:
            return None

        target = self.find_lookahead_point()
        if target is None:
            self.active = False
            return 0.0, 0.0

        # End of the current same-gear segment: finished, or on to the next.
        seg_end = self._seg_ends[self._seg]
        final = self._seg == len(self._seg_ends) - 1
        if self._search_idx >= seg_end and self._segment_end_reached(seg_end, final):
            if final:
                self.active = False
                return 0.0, 0.0
            self._seg += 1
            self._search_idx = seg_end + 1
            target = self.find_lookahead_point()

        target_x, target_y, gear = target
        # Drive toward where the target actually IS, not blindly in the
        # planned gear. After a forward segment ends a little off the line,
        # the first REVERSE waypoint can already be in front of the car (and
        # vice versa); the planned gear then points straight away from it,
        # calculate_pure_pursuit() refuses to steer, and the car drives off
        # forever - seen in sim 2026-09-24 at the end of leg 1 (forward then
        # 5 reverse points), reversing 13 m out of the arena.
        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        local_x = dx * math.cos(self.current_pose[2]) + dy * math.sin(self.current_pose[2])
        if gear >= 0 and local_x < 0.0:
            gear = -1
        elif gear < 0 and local_x > 0.0:
            gear = 1
        self.last_target = (target_x, target_y, gear)
        steering_angle = self.calculate_pure_pursuit((target_x, target_y), gear)
        speed = self.target_speed if gear >= 0 else -self.target_speed
        # ackermann_steering_controller's /cmd_vel takes body yaw rate
        # (angular.z = omega), not the steering angle itself - convert via
        # the standard bicycle-model relation omega = v*tan(delta)/L. Using
        # the signed `speed` here (not self.target_speed) is what makes
        # reverse segments turn the correct way instead of mirroring a
        # forward turn at negative speed.
        angular_z = speed * math.tan(steering_angle) / self.wheelbase
        return speed, angular_z


class PurePursuitFollower(Node):
    """Standalone node wrapper - see module docstring. Not part of the
    default launch graph; task1_runner.py uses PurePursuitController
    directly instead so there's only one /cmd_vel publisher during Task 1.

    FRAME SCOPE: this node is NOT arena-frame aware. `odom_cb` below feeds
    `/odometry/filtered` straight into the controller, and that message reports in
    the dead-reckoning frame, so the pose this node tracks against is a
    dead-reckoning pose. Paths given to `set_path` must therefore be expressed in
    that same dead-reckoning frame - an arena-coordinate path (what task 1's
    planner produces) would be tracked with a constant rotation and offset equal
    to the robot's start pose. task1_runner.py handles that case by looking the
    transform up in TF before it calls `update_pose`; if this standalone node ever
    needs to drive an arena-frame path, it needs the same treatment.

    `PurePursuitController` itself is frame-agnostic: it only requires that the
    pose and the path share one frame, whichever that is."""

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
