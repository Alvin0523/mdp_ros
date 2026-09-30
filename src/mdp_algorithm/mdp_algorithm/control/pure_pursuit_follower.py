#!/usr/bin/env python3
"""
Pure pursuit path follower for the Ackermann car (ROS-free).

Give it a path of (x, y, theta, gear) points and a stream of poses; each tick
compute_cmd() returns the /cmd_vel (linear.x, angular.z) to send. task1_runner
owns the ROS side (pose from TF, publishing).

How it drives:
  * The path is split at every direction change (cusp) into same-gear
    segments, driven one at a time.
  * It steers at the first path point at least lookahead_dist away, searching
    forward only (never re-targets a point already passed). Near the end of a
    segment the target is the segment's last point.
  * A cusp is finished when the car reaches it (within cusp_tolerance), is past
    it, or it is no longer ahead of the car - then the next segment starts (as
    Nav2's Regulated Pure Pursuit: reach the cusp, THEN change gear; switching
    early started the reverse arc in the wrong place).
  * The final point is reached within xy_goal_tolerance (Nav2 goal checker);
    once the car is past it anyway, it stops there (no replanning yet - driving
    on only dithers beside the goal).
  * Steering is clamped PER SIDE to the measured wheel limits (43 deg left,
    32.5 deg right) and converted to yaw rate with the bicycle model.

Settings default to robot.* / follower.* in mdp_bringup/config/navigation.yaml
(utils/params.py, read when the controller is built); any argument overrides.
Why lookahead 0.10 m: pure pursuit tracks a curve only when the lookahead is
well below its radius (0.25 m cut the corners and hit obstacle 1, 2026-09-17).
"""

import math
from typing import List, Optional, Tuple

from ..utils import params as planner_params

Pose = Tuple[float, float, float]
DensePose = Tuple[float, float, float, int]   # (x, y, theta, gear): gear +1 forward, -1 reverse


def yaw_from_quaternion(q) -> float:
    """Yaw of a geometry_msgs/Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PurePursuitController:
    def __init__(self, wheelbase: Optional[float] = None, lookahead_dist: Optional[float] = None,
                 max_steering_left: Optional[float] = None, max_steering_right: Optional[float] = None,
                 target_speed: Optional[float] = None, goal_tolerance: Optional[float] = None,
                 cusp_tolerance: Optional[float] = None,
                 params: Optional[planner_params.PlannerParams] = None):
        p = params or planner_params.ACTIVE

        def pick(value, default):
            return default if value is None else value

        self.wheelbase = pick(wheelbase, p.wheelbase)                                # m
        self.lookahead_dist = pick(lookahead_dist, p.lookahead_dist)                 # m
        self.max_steering_left = pick(max_steering_left, p.steering_limit_left)      # rad, > 0
        self.max_steering_right = pick(max_steering_right, p.steering_limit_right)   # rad, > 0
        self.target_speed = pick(target_speed, p.desired_linear_vel)                 # m/s
        self.goal_tolerance = pick(goal_tolerance, p.xy_goal_tolerance)              # m
        self.cusp_tolerance = pick(cusp_tolerance, p.cusp_tolerance)                 # m

        self.path: List[DensePose] = []
        self.current_pose: Pose = (0.0, 0.0, 0.0)
        self.active = False
        self.last_target = None        # (x, y, gear) last steered at - for /run_status
        self._search_idx = 0           # never searches behind this point
        self._seg_ends: List[int] = []  # last path index of each same-gear segment
        self._seg = 0                  # segment being driven

    def set_path(self, path_waypoints: List[DensePose]) -> None:
        """(x, y, theta, gear) points; a point without gear is driven forward."""
        self.path = [p if len(p) == 4 else (p[0], p[1], p[2], 1) for p in path_waypoints]
        self.active = bool(self.path)
        self.last_target = None
        self._search_idx = 0
        self._seg_ends = [i - 1 for i in range(1, len(self.path))
                          if self.path[i][3] != self.path[i - 1][3]] + [len(self.path) - 1]
        self._seg = 0

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.current_pose = (x, y, yaw)

    def is_done(self) -> bool:
        return not self.active

    def find_lookahead_point(self) -> Optional[Tuple[float, float, int]]:
        """(x, y, gear) of the first point in this segment at least
        lookahead_dist away, else the segment's last point."""
        if not self.path:
            return None
        seg_end = self._seg_ends[self._seg]
        x, y, _ = self.current_pose
        for i in range(self._search_idx, seg_end + 1):
            px, py, _, gear = self.path[i]
            if math.hypot(px - x, py - y) >= self.lookahead_dist:
                self._search_idx = i
                return px, py, gear
        self._search_idx = seg_end
        px, py, _, gear = self.path[seg_end]
        return px, py, gear

    def _segment_end_reached(self, idx: int, final: bool) -> bool:
        """Is the segment ending at path[idx] finished? (see module docstring)"""
        px, py, ptheta, gear = self.path[idx]
        x, y, yaw = self.current_pose
        dx, dy = x - px, y - py
        direction = 1.0 if gear >= 0 else -1.0
        past = direction * (dx * math.cos(ptheta) + dy * math.sin(ptheta)) > 0.0
        ahead = -direction * (dx * math.cos(yaw) + dy * math.sin(yaw))   # point ahead of the car (+)
        # "Passed" / "no longer ahead" only mean something when the car is roughly
        # lined up with that point's heading. Halfway round a U-turn the end of
        # the path (facing the other way) is briefly behind the car too, and
        # counting that ended task 2's loop round obstacle 2 early (2026-09-30).
        aligned = math.cos(yaw - ptheta) > 0.5          # within 60 deg
        if not final:
            reached = direction * (dx * math.cos(ptheta) + dy * math.sin(ptheta)) >= -self.cusp_tolerance
            return reached or (aligned and ahead <= 0.0)
        return math.hypot(dx, dy) <= self.goal_tolerance or (aligned and (past or ahead <= 0.0))

    def calculate_pure_pursuit(self, target_point: Tuple[float, float], gear: int = 1) -> float:
        """Steering angle (rad, + left) onto the arc through target_point. The
        arc is the same circle whichever way it is driven; the signed speed in
        compute_cmd() turns it the right way in reverse. 0 if the target is on
        the wrong side of the car for this gear."""
        x, y, yaw = self.current_pose
        dx, dy = target_point[0] - x, target_point[1] - y
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if (local_x <= 0.0) if gear >= 0 else (local_x >= 0.0):
            return 0.0
        curvature = 2.0 * local_y / self.lookahead_dist ** 2
        steering = math.atan(self.wheelbase * curvature)
        return max(-self.max_steering_right, min(self.max_steering_left, steering))

    def compute_cmd(self) -> Optional[Tuple[float, float]]:
        """(linear_x, angular_z) for this tick; (0, 0) when the path is done,
        None when there is no path. Done only once the whole path has been
        consumed - not merely when the car is near the end, which used to skip
        a final reverse (2026-09-17)."""
        if not self.active or not self.path:
            return None
        target = self.find_lookahead_point()

        seg_end = self._seg_ends[self._seg]
        final = self._seg == len(self._seg_ends) - 1
        if self._search_idx >= seg_end and self._segment_end_reached(seg_end, final):
            if final:
                self.active = False
                return 0.0, 0.0
            self._seg += 1
            self._search_idx = seg_end + 1
            target = self.find_lookahead_point()

        # Drive toward where the target actually is, not blindly in the planned
        # gear: just after a cusp the next segment's first point can be on the
        # other side of the car (else it drives off - 2026-09-24, 13 m).
        tx, ty, gear = target
        x, y, yaw = self.current_pose
        local_x = (tx - x) * math.cos(yaw) + (ty - y) * math.sin(yaw)
        if gear >= 0 and local_x < 0.0:
            gear = -1
        elif gear < 0 and local_x > 0.0:
            gear = 1
        self.last_target = (tx, ty, gear)

        steering = self.calculate_pure_pursuit((tx, ty), gear)
        speed = self.target_speed if gear >= 0 else -self.target_speed
        # ackermann_steering_controller takes yaw rate: w = v * tan(delta) / L.
        return speed, speed * math.tan(steering) / self.wheelbase
