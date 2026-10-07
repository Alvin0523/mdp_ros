#!/usr/bin/env python3
"""
Path follower for the Ackermann car (ROS-free): pure pursuit, feedback or LQR
steering (follower.path_tracking; task 1 runs lqr).

Give it a path of (x, y, theta, gear) points and a stream of poses; each tick
compute_cmd() returns the /cmd_vel (linear.x, angular.z) to send. task1_runner
owns the ROS side (pose from TF, publishing).

How it drives:
  * The path is split at every direction change (cusp) into same-gear
    segments, driven one at a time.
  * It steers at the first path point at least lookahead_dist away, searching
    forward only (never re-targets a point already passed). Near the end of a
    segment the target is beyond its last point, along the end heading.
  * A cusp is finished when the car reaches it (within cusp_tolerance), is past
    it, or it is no longer ahead of the car - then the next segment starts (as
    Nav2's Regulated Pure Pursuit: reach the cusp, THEN change gear; switching
    early started the reverse arc in the wrong place).
  * The final point is reached like a cusp - driven up to (cusp_tolerance) and
    within xy_goal_tolerance of it; once the car is past it anyway, it stops
    there (driving on only dithers beside the goal). Near the end of a segment
    it steers at a point beyond the end along the end heading, so it arrives
    lined up (task1_runner replans once if it still arrives too far off).
  * Steering is clamped PER SIDE to the measured wheel limits (43 deg left,
    32.5 deg right) and converted to yaw rate with the bicycle model.
  * The pose comes in pose_latency late (EKF + transport: ~60 ms in Gazebo,
    2026-09-30 - the car was ~2 cm further along than its pose said at 0.3 m/s),
    so it is moved forward by the last command over that time first.
  * Steering (tracking): 'pure_pursuit' steers onto the arc through the
    lookahead point - it cuts inside curves by ~lookahead^2 / (2 radius).
    'feedback' (rear-axle path feedback, e.g. PythonRobotics
    rear_wheel_feedback) steers by the planned path's own curvature at the
    nearest point (looked up feedforward_preview_time ahead, for the steering
    lag) and corrects the sideways error and heading error:
        w = |v| k_path - k_theta |v| e_theta - k_e v e sin(e_theta)/e_theta
    (a Lyapunov-stable law, forward and reverse). Tried for task 1 because pure
    pursuit drifted ~5 cm off its paths at every speed (sim, 2026-09-30).
    'lqr' holds the same nearest-point errors plus the wheel angle in a small
    model - sideways error e, heading error, wheel angle lagging its command
    with steering_time_constant (Gazebo: 90% of a step in ~0.5 s) - and steers
    with the optimal gain for it (discrete LQR, solved once per speed). Knowing
    the lag, it starts turning before a curve instead of after. The wheel angle
    is not measured: it is the model's estimate from the commands sent.
  * Speed (regulate=True, task 1 - as Nav2's Regulated Pure Pursuit): target_speed
    on straights, less where it is tight - the path's sharpest curve in the next
    stretch caps it at sqrt(max_lateral_accel x radius), reversing at
    max_reverse_linear_vel, and it slows over the last
    approach_velocity_scaling_dist before every cusp and the goal. The lookahead
    grows with the speed actually driven, not the target. Why: at 0.4 m/s flat
    out the car ran 5.6 cm off its path reversing into a checkpoint and touched
    the block (sim, 2026-09-30) - the steering takes ~0.66 s to swing over and a
    long lookahead cuts corners. task2_runner sets its own speed (regulate=False).

Settings default to robot.* / follower.* in mdp_bringup/config/navigation.yaml
(utils/params.py, read when the controller is built); any argument overrides.
Why lookahead 0.10 m: pure pursuit tracks a curve only when the lookahead is
well below its radius (0.25 m cut the corners and hit obstacle 1, 2026-09-17).
"""

import math
from typing import List, Optional, Tuple

from ..utils import params as planner_params

CONTROL_DT = 0.05             # s, compute_cmd() is called at 20 Hz (task1_runner, goto)
LOOKAHEAD_REF_SPEED = 0.2    # m/s at which lookahead_dist applies (it was tuned there)
GEAR_FLIP_M = 0.03           # m the target must be behind (ahead) before driving the other gear

Pose = Tuple[float, float, float]
DensePose = Tuple[float, float, float, int]   # (x, y, theta, gear): gear +1 forward, -1 reverse


def yaw_from_quaternion(q) -> float:
    """Yaw of a geometry_msgs/Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PathFollower:
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
        self.max_path_error = p.max_path_error                                       # m, see path_error()
        self.velocity_scaled_lookahead = p.use_velocity_scaled_lookahead_dist
        self.regulate = p.use_regulated_linear_velocity_scaling
        self.max_lateral_accel = p.max_lateral_accel                                 # m/s^2
        self.min_regulated_speed = p.regulated_linear_scaling_min_speed              # m/s
        self.max_reverse_speed = p.max_reverse_linear_vel                            # m/s
        self.approach_dist = p.approach_velocity_scaling_dist                        # m
        self.min_approach_speed = p.min_approach_linear_velocity                     # m/s
        self.speed = 0.0               # |speed| of the last command
        self.pose_latency = p.pose_latency                                           # s
        self._last_cmd = (0.0, 0.0)    # (v, w) sent last tick, for the latency prediction
        self.tracking = p.path_tracking                                              # pure_pursuit | feedback
        self.k_e = p.feedback_k_e                                                    # 1/m^2
        self.k_theta = p.feedback_k_theta                                            # 1/m
        self.preview_time = p.feedforward_preview_time                               # s
        self.lqr_weights = (p.lqr_q_lateral, p.lqr_q_heading, p.lqr_q_steer, p.lqr_r)
        self.steer_tau = p.steering_time_constant                                    # s
        self.steer_est = 0.0           # rad, the model's wheel angle (lqr)
        self._lqr_cache = {}

        self.path: List[DensePose] = []
        self.current_pose: Pose = (0.0, 0.0, 0.0)
        self.measured_pose: Pose = (0.0, 0.0, 0.0)
        self.active = False
        self.last_target = None        # (x, y, gear) last steered at - for /run_status
        self._search_idx = 0           # never searches behind this point
        self._seg_ends: List[int] = []  # last path index of each same-gear segment
        self._seg = 0                  # segment being driven
        self._near_idx = 0             # path point nearest the car (only moves forward)
        self.speed_cap = math.inf      # m/s, set by the caller (task 1: creep onto the block)

    def on_last_segment(self) -> bool:
        """Driving the path's last same-gear stretch (the one into the goal)."""
        return self.active and self._seg == len(self._seg_ends) - 1

    def set_path(self, path_waypoints: List[DensePose]) -> None:
        """(x, y, theta, gear) points; a point without gear is driven forward."""
        self.path = [p if len(p) == 4 else (p[0], p[1], p[2], 1) for p in path_waypoints]
        self.active = bool(self.path)
        self.last_target = None
        self._search_idx = 0
        self._seg_ends = [i - 1 for i in range(1, len(self.path))
                          if self.path[i][3] != self.path[i - 1][3]] + [len(self.path) - 1]
        self._seg = 0
        self._near_idx = 0
        self._path_start = self.current_pose[:2]   # where the car was: the path's first point is a step ahead
        self.speed = 0.0
        self.steer_est = 0.0
        self._last_cmd = (0.0, 0.0)

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.measured_pose = (x, y, yaw)
        self.current_pose = self.predicted_pose()

    def predicted_pose(self) -> Pose:
        """The measured pose moved on by the last command over pose_latency.

        lqr: with the turn rate of the model's wheel angle (steer_est), not the
        commanded one. The wheels lag the command (~0.2 s), so predicting with the
        command made a flipped command flip the predicted heading too, which the
        LQR answered by flipping back - lock-to-lock about 10 times a second
        (real car 2026-10-01: 35-60% of steering commands reversed; a simulation
        with the measured lag and delay gave 78%, and 4% predicting this way with
        lqr_r 2.0)."""
        x, y, yaw = self.measured_pose
        v, w = self._last_cmd
        if self.tracking == 'lqr':
            w = v * math.tan(self.steer_est) / self.wheelbase
        dt = self.pose_latency
        if dt <= 0.0 or (v == 0.0 and w == 0.0):
            return x, y, yaw
        mid = yaw + 0.5 * w * dt
        return x + v * dt * math.cos(mid), y + v * dt * math.sin(mid), yaw + w * dt

    def is_done(self) -> bool:
        return not self.active

    def path_error(self) -> float:
        """m from the car (rear axle) to the LINE of the segment being driven
        (near the nearest point). Callers stop and replan past max_path_error:
        off the path the car is somewhere the planner never checked for blocks
        (real car 2026-10-01: 14 cm off on a long S-curve, it drove into two).

        The line, not the points: they are 2.5-5 cm apart and the first one is a
        planner step ahead of the start, so the distance to the nearest point
        read 5-6 cm for a car sitting on the path and stopped good runs."""
        if not self.active or not self.path:
            return 0.0
        seg_end = self._seg_ends[self._seg]
        start = self._seg_start()
        i = self._update_near(seg_end)
        x, y, _ = self.current_pose
        pts = [p[:2] for p in self.path[max(start, i - 4):min(seg_end, i + 4) + 1]]
        if max(start, i - 4) == start:      # the line into the segment's first point
            pts.insert(0, self.path[start - 1][:2] if start > 0 else self._path_start)
        best = min(math.hypot(px - x, py - y) for px, py in pts)
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            best = min(best, math.hypot(ax + t * dx - x, ay + t * dy - y))
        return best

    def lookahead(self) -> float:
        """lookahead_dist at LOOKAHEAD_REF_SPEED, growing with the speed driven."""
        if not self.velocity_scaled_lookahead:
            return self.lookahead_dist
        return self.lookahead_dist * max(1.0, self.speed / LOOKAHEAD_REF_SPEED)

    def _update_near(self, seg_end: int) -> int:
        x, y, _ = self.current_pose
        i = max(self._near_idx, self._seg_start())
        best = math.hypot(self.path[i][0] - x, self.path[i][1] - y)
        for j in range(i + 1, seg_end + 1):
            d = math.hypot(self.path[j][0] - x, self.path[j][1] - y)
            if d > best + 0.05:        # past the nearest (don't jump to a later pass nearby)
                break
            if d <= best:
                i, best = j, d
        self._near_idx = i
        return i

    def _seg_start(self) -> int:
        return self._seg_ends[self._seg - 1] + 1 if self._seg > 0 else 0

    def regulated_speed(self, gear: int) -> float:
        """|speed| for this tick (module docstring)."""
        v = self.target_speed
        if not self.regulate:
            return v
        if gear < 0:
            v = min(v, self.max_reverse_speed)
        seg_end = self._seg_ends[self._seg]
        i = self._update_near(seg_end)
        # Sharpest curve over the stretch the car covers before it could slow
        # down, and how much of this segment is left (to the cusp / goal).
        preview = self.lookahead() + v * v / (2.0 * self.max_lateral_accel) + 0.05
        s, kappa = 0.0, 0.0
        for j in range(i, seg_end):
            a, b = self.path[j], self.path[j + 1]
            ds = math.hypot(b[0] - a[0], b[1] - a[1])
            if s < preview and ds > 1e-4:
                dth = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
                kappa = max(kappa, abs(dth) / ds)
            s += ds
        left = s
        if kappa > 1e-6:
            v = min(v, max(self.min_regulated_speed, math.sqrt(self.max_lateral_accel / kappa)))
        if left < self.approach_dist:
            v = min(v, max(self.min_approach_speed, self.target_speed * left / self.approach_dist))
        return v

    def find_lookahead_point(self) -> Optional[Tuple[float, float, int]]:
        """(x, y, gear) of the first point in this segment at least
        lookahead_dist away. Within lookahead_dist of the segment's last point,
        a point that far ahead on the line through it along its heading.

        That extension (2026-10-01): aiming at the end point itself makes pure
        pursuit arrive at the right place pointing anywhere - with the lookahead
        longer than a short final piece after a gear change, task 1 arrived 61 deg
        off. Aiming past it along the end heading lines the car up on the way in."""
        if not self.path:
            return None
        seg_end = self._seg_ends[self._seg]
        x, y, _ = self.current_pose
        look = self.lookahead()
        for i in range(self._search_idx, seg_end + 1):
            px, py, _, gear = self.path[i]
            if math.hypot(px - x, py - y) >= look:
                self._search_idx = i
                return px, py, gear
        self._search_idx = seg_end
        px, py, ptheta, gear = self.path[seg_end]
        extra = look - math.hypot(px - x, py - y)
        direction = 1.0 if gear >= 0 else -1.0
        return px + direction * extra * math.cos(ptheta), py + direction * extra * math.sin(ptheta), gear

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
        reached = direction * (dx * math.cos(ptheta) + dy * math.sin(ptheta)) >= -self.cusp_tolerance
        if not final:
            return reached or (aligned and ahead <= 0.0)
        # The goal like a cusp: drive up to it (cusp_tolerance), within
        # goal_tolerance of it sideways. Any point within goal_tolerance used to
        # count, so the car stopped ~5 cm short, still turning onto the final
        # heading (real car 2026-10-01: arrived 7-14 deg off, then rolled on).
        return (reached and math.hypot(dx, dy) <= self.goal_tolerance) or (aligned and (past or ahead <= 0.0))

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
        curvature = 2.0 * local_y / self.lookahead() ** 2
        steering = math.atan(self.wheelbase * curvature)
        return max(-self.max_steering_right, min(self.max_steering_left, steering))

    def _path_curvature(self, i: int, seg_end: int) -> float:
        """Signed d(heading)/d(distance travelled) of the path at point i."""
        j = min(i + 1, seg_end)
        k = max(j - 2, self._seg_start())
        a, b = self.path[k], self.path[j]
        ds = math.hypot(b[0] - a[0], b[1] - a[1])
        if ds < 1e-4:
            return 0.0
        return math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2])) / ds

    def feedback_yaw_rate(self, v: float) -> float:
        """Yaw rate from the path feedback law (module docstring); v signed."""
        seg_end = self._seg_ends[self._seg]
        i = self._update_near(seg_end)
        px, py, pth, _ = self.path[i]
        x, y, yaw = self.current_pose
        e = -(x - px) * math.sin(pth) + (y - py) * math.cos(pth)          # + = car left of the path
        e_th = math.atan2(math.sin(yaw - pth), math.cos(yaw - pth))
        # Curvature where the car will be once the steering has swung over.
        j, travelled = i, 0.0
        while j < seg_end and travelled < abs(v) * self.preview_time:
            travelled += math.hypot(self.path[j + 1][0] - self.path[j][0], self.path[j + 1][1] - self.path[j][1])
            j += 1
        k_path = self._path_curvature(j, seg_end)
        sinc = math.sin(e_th) / e_th if abs(e_th) > 1e-6 else 1.0
        return abs(v) * k_path - self.k_theta * abs(v) * e_th - self.k_e * v * e * sinc

    def _lqr_gain(self, v: float, dt: float):
        key = (round(v, 2), dt, self.lqr_weights, self.steer_tau)
        if key not in self._lqr_cache:
            import numpy as np
            a = dt / max(self.steer_tau, dt)
            A = np.array([[1.0, v * dt, 0.0], [0.0, 1.0, v * dt / self.wheelbase], [0.0, 0.0, 1.0 - a]])
            B = np.array([[0.0], [0.0], [a]])
            q_e, q_th, q_d, r = self.lqr_weights
            Q, R = np.diag([q_e, q_th, q_d]), np.array([[r]])
            P = Q.copy()
            for _ in range(300):       # discrete Riccati equation, by iteration
                K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
                P = Q + A.T @ P @ (A - B @ K)
            self._lqr_cache[key] = K[0]
        return self._lqr_cache[key]

    def lqr_steering(self, v: float, dt: float = CONTROL_DT) -> float:
        """Wheel angle command (rad) from the LQR (module docstring); v signed."""
        seg_end = self._seg_ends[self._seg]
        i = self._update_near(seg_end)
        px, py, pth, _ = self.path[i]
        x, y, yaw = self.current_pose
        e = -(x - px) * math.sin(pth) + (y - py) * math.cos(pth)
        e_th = math.atan2(math.sin(yaw - pth), math.cos(yaw - pth))
        j, travelled = i, 0.0
        while j < seg_end and travelled < abs(v) * self.preview_time:
            travelled += math.hypot(self.path[j + 1][0] - self.path[j][0], self.path[j + 1][1] - self.path[j][1])
            j += 1
        # The wheel angle the path needs there: heading change per metre driven,
        # turned into a wheel angle; in reverse the same curve needs the opposite.
        k = self._path_curvature(j, seg_end)
        ref = math.atan(self.wheelbase * k) * (1.0 if v > 0 else -1.0)
        kv = self._lqr_gain(v, dt)
        cmd = ref - (kv[0] * e + kv[1] * e_th + kv[2] * (self.steer_est - ref))
        cmd = max(-self.max_steering_right, min(self.max_steering_left, cmd))
        self.steer_est += (cmd - self.steer_est) * min(1.0, dt / max(self.steer_tau, dt))
        return cmd

    def compute_cmd(self) -> Optional[Tuple[float, float]]:
        """(linear_x, angular_z) for this tick; (0, 0) when the path is done,
        None when there is no path. Done only once the whole path has been
        consumed - not merely when the car is near the end, which used to skip
        a final reverse (2026-09-17)."""
        if not self.active or not self.path:
            return None
        self.current_pose = self.predicted_pose()
        target = self.find_lookahead_point()

        seg_end = self._seg_ends[self._seg]
        final = self._seg == len(self._seg_ends) - 1
        if self._search_idx >= seg_end and self._segment_end_reached(seg_end, final):
            if final:
                self.active = False
                self._last_cmd = (0.0, 0.0)
                return 0.0, 0.0
            self._seg += 1
            self._search_idx = seg_end + 1
            target = self.find_lookahead_point()

        # Drive toward where the target actually is, not blindly in the planned
        # gear: just after a cusp the next segment's first point can be on the
        # other side of the car (else it drives off - 2026-09-24, 13 m).
        # Only when it is CLEARLY on the other side (GEAR_FLIP_M): a target
        # almost beside the car flipped the gear every tick - forward, reverse,
        # forward at full lock, going nowhere (real car 2026-10-01).
        tx, ty, gear = target
        x, y, yaw = self.current_pose
        local_x = (tx - x) * math.cos(yaw) + (ty - y) * math.sin(yaw)
        if gear >= 0 and local_x < -GEAR_FLIP_M:
            gear = -1
        elif gear < 0 and local_x > GEAR_FLIP_M:
            gear = 1
        self.last_target = (tx, ty, gear)

        self.speed = min(self.regulated_speed(gear), self.speed_cap)
        speed = self.speed if gear >= 0 else -self.speed
        if self.tracking == 'lqr' and abs(speed) > 1e-3:
            steering = self.lqr_steering(speed)
        elif self.tracking == 'feedback' and abs(speed) > 1e-3:
            w = self.feedback_yaw_rate(speed)
            steering = math.atan(self.wheelbase * w / speed)
            steering = max(-self.max_steering_right, min(self.max_steering_left, steering))
        else:
            steering = self.calculate_pure_pursuit((tx, ty), gear)
        # ackermann_steering_controller takes yaw rate: w = v * tan(delta) / L.
        self._last_cmd = (speed, speed * math.tan(steering) / self.wheelbase)
        return self._last_cmd
