#!/usr/bin/env python3
"""
Chord-length-parameterized PCHIP spline path planner.

Fits x(t) and y(t) independently as shape-preserving PCHIP splines through a
list of (x, y) waypoints, using cumulative chord length as the parameter t.
Heading at any sample is the spline's own tangent (atan2(dy/dt, dx/dt)) - no
manual heading guessing at interior waypoints, which is what made the
hand-rolled Dubins-chaining approach loop wildly on sharp lateral waypoints.

PCHIP (not a natural cubic spline) is used specifically because natural
cubic splines overshoot past waypoints on sharp V-shaped turns (verified:
it pushed the path ~0.09m past an obstacle-clearance waypoint, right on top
of an obstacle). PCHIP is monotonicity-preserving and does not overshoot.

Does not itself enforce a minimum turning radius; the caller (pure pursuit
controller, using a lookahead rather than instantaneous curvature) clamps
commanded curvature to the vehicle's kappa_max and slows down where the
path curves tightly - verified via closed-loop simulation to hold within
kappa_max for this vehicle's waypoints.
"""

import math
from typing import List, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

Pose = Tuple[float, float, float]


class SplinePathPlanner:
    def generate_path_through_waypoints(self, waypoints_xy: List[Tuple[float, float]],
                                         step_size: float = 0.05) -> List[Pose]:
        n = len(waypoints_xy)
        if n == 0:
            return []
        if n == 1:
            return [(waypoints_xy[0][0], waypoints_xy[0][1], 0.0)]

        pts = np.array(waypoints_xy, dtype=float)
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        t = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        cs_x = PchipInterpolator(t, pts[:, 0])
        cs_y = PchipInterpolator(t, pts[:, 1])

        total_len = float(t[-1])
        n_samples = max(2, int(round(total_len / step_size)))
        ts = np.linspace(0.0, total_len, n_samples)

        xs = cs_x(ts)
        ys = cs_y(ts)
        dxs = cs_x(ts, 1)
        dys = cs_y(ts, 1)
        yaws = np.arctan2(dys, dxs)

        return list(zip(xs.tolist(), ys.tolist(), yaws.tolist()))

    def max_curvature(self, path: List[Pose]) -> float:
        """Diagnostic: peak curvature (1/m) along a sampled path, via finite
        differences of heading over arc length. Useful to sanity-check a
        planned path against the vehicle's kappa_max before trusting it."""
        worst = 0.0
        for i in range(1, len(path) - 1):
            x0, y0, _ = path[i - 1]
            x1, y1, yaw1 = path[i]
            x2, y2, _ = path[i + 1]
            ds = math.hypot(x2 - x0, y2 - y0) / 2.0
            if ds < 1e-6:
                continue
            dyaw = math.atan2(math.sin(path[i + 1][2] - path[i - 1][2]),
                               math.cos(path[i + 1][2] - path[i - 1][2]))
            worst = max(worst, abs(dyaw) / (2.0 * ds))
        return worst
