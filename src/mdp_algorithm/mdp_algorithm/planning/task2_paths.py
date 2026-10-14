"""
Task 2's clean geometric legs (ROS-free): what task2_runner drives, and what the
CP1 -> CP2 route table (tools/task2_table.py) is built from offline.

  smooth_s_path()  start -> goal with the same heading, a sideways offset between
                   them: straight, an S whose curvature rises and falls smoothly
                   (a quintic, zero curvature at both ends), straight again. The
                   steering never has to jump - it swings over gradually, and the
                   car leaves the S already pointing straight (2026-10-14: the
                   arc - straight - arc S jumped from full lock to straight at the
                   end of each arc; at 0.7 m/s the car overshot the straighten-up
                   into checkpoint 1 and arrived 30 deg off).
  arc_s_path()     the older S: straight, an arc out, a straight diagonal, an arc
                   back - tighter, kept for where no smooth S fits.
  lined_up_path()  either S, then `straight` m straight into the goal.
  round_path()     half a circle round the back of the bar (checkpoint 2 -> 3).

Poses are (x, y, theta, gear) every `step` m, metres / radians, gear +1 (task 2
drives forward only). `clear()` checks a path against a Costmap (cm), with room
to spare either side.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Pose = Tuple[float, float, float]
DensePose = Tuple[float, float, float, int]


@dataclass(frozen=True)
class Shapes:
    """What the shapes may do: the planned turning circles (the minimum x
    planner.turning_radius_margin, per side) and the speed profile they are
    timed with (task2_runner straight_speed, max_lateral_accel)."""
    r_left: float               # m
    r_right: float              # m
    straight_speed: float       # m/s
    max_lateral_accel: float    # m/s^2

    @classmethod
    def from_params(cls, car, straight_speed: float, max_lateral_accel: float) -> 'Shapes':
        """car: a PlannerParams (utils/params.py)."""
        k = car.turning_radius_margin
        return cls(car.minimum_turning_radius_left * k, car.minimum_turning_radius_right * k,
                   float(straight_speed), float(max_lateral_accel))


# ------------------------------------------------------------- helpers ----

def to_world(start: Pose, local: List[DensePose]) -> List[DensePose]:
    """Points in start's frame (start at the origin facing +x) -> the map frame."""
    c, s = math.cos(start[2]), math.sin(start[2])
    return [(start[0] + c * x - s * y, start[1] + s * x + c * y, start[2] + th, g) for x, y, th, g in local]


def to_local(start: Pose, pose: Pose) -> Pose:
    """A map-frame pose in start's frame."""
    c, s = math.cos(start[2]), math.sin(start[2])
    dx, dy = pose[0] - start[0], pose[1] - start[1]
    return c * dx + s * dy, -s * dx + c * dy, pose[2] - start[2]


def clear(path: List[DensePose], costmap, margin: float = 0.04) -> bool:
    """Every (3rd) pose collision-free, and still so shifted `margin` to either
    side: room for the car running a few cm off the line."""
    for x, y, th, _ in path[::3]:
        nx, ny = -math.sin(th) * margin, math.cos(th) * margin
        for ox, oy in ((0.0, 0.0), (nx, ny), (-nx, -ny)):
            if costmap.in_collision((x + ox) * 100.0, (y + oy) * 100.0, th):
                return False
    return True


def curvatures(path: List[DensePose]) -> List[float]:
    """Signed heading change per metre between consecutive points (+ = left)."""
    out = []
    for p0, p1 in zip(path, path[1:]):
        ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        dth = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        out.append(dth / ds if ds > 1e-6 else 0.0)
    return out


def drive_time(path: List[DensePose], shapes: Shapes) -> float:
    """Seconds to drive `path` at the speed profile: straight_speed, curves at
    sqrt(max_lateral_accel x radius)."""
    t = 0.0
    for (p0, p1), kappa in zip(zip(path, path[1:]), curvatures(path)):
        ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if ds < 1e-6:
            continue
        kappa = abs(kappa)
        v = min(shapes.straight_speed, math.sqrt(shapes.max_lateral_accel / kappa)) if kappa > 1e-6 \
            else shapes.straight_speed
        t += ds / v
    return t


def straight(a: Pose, b: Pose, step: float = 0.02, include_start: bool = False) -> List[DensePose]:
    """a -> b in a line at b's heading, every ~step m (b included)."""
    n = max(1, int(round(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, b[2], 1)
            for k in range(0 if include_start else 1, n + 1)]


# ---------------------------------------------------------- smooth S ----

S_LENGTHS = np.arange(0.30, 2.01, 0.10)     # m along the start heading the S may take (every 10 cm: the
                                            # search ran ~0.3 s for the home leg at 5 cm on a laptop)
S_EARLIEST_STEP = 0.05                      # m: where the S starts, tried from the start on


def quintic_s(length: float, offset: float, step: float = 0.02) -> List[DensePose]:
    """The S alone, from the origin facing +x to (length, offset) facing +x:
    y(x) = offset * (10u^3 - 15u^4 + 6u^5), u = x / length - heading and
    curvature both zero at either end."""
    xs = np.linspace(0.0, length, max(50, int(length / 0.0025)))
    u = xs / length
    y = offset * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
    dy = offset * (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / length
    th = np.arctan(dy)
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(y)))))
    targets = np.arange(step, s[-1], step)
    out = [(float(np.interp(t, s, xs)), float(np.interp(t, s, y)), float(np.interp(t, s, th)), 1) for t in targets]
    return out + [(float(length), float(offset), 0.0, 1)]


def s_fits_turning_circles(length: float, offset: float, shapes: Shapes) -> bool:
    """The S's sharpest curvature, each way, within the planned turning circles:
    |y''| / (1 + y'^2)^1.5; the first half curves toward the offset's side."""
    u = np.linspace(0.0, 1.0, 201)
    d1 = offset * (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / length
    d2 = offset * (60 * u - 180 * u ** 2 + 120 * u ** 3) / length ** 2
    k = d2 / (1.0 + d1 ** 2) ** 1.5
    return k.max() <= 1.0 / shapes.r_left + 1e-9 and -k.min() <= 1.0 / shapes.r_right + 1e-9


def clear_run(origin: Pose, length: float, costmap, margin: float, backward: bool = False,
              step: float = 0.02) -> float:
    """How far (m, up to length) a straight from origin along its heading - or back
    against it - stays clear: the straights before and after an S are checked once,
    not once per S tried."""
    c, s = math.cos(origin[2]), math.sin(origin[2])
    sign = -1.0 if backward else 1.0
    d = 0.0
    while d <= length + 1e-9:
        pose = (origin[0] + sign * d * c, origin[1] + sign * d * s, origin[2], 1)
        if not clear([pose], costmap, margin):
            return max(0.0, d - step)
        d += step
    return length


def smooth_s_path(start: Pose, goal: Pose, costmap, shapes: Shapes, margin: float = 0.04,
                  step: float = 0.02) -> Optional[List[DensePose]]:
    """start -> goal (same heading) as the FASTEST smooth S that fits: for each
    S length the earliest start that is clear by `margin`. None if none fits."""
    gx, gy, _ = to_local(start, goal)
    if gx <= 0.0:
        return None
    if abs(gy) < 0.005:
        path = to_world(start, straight((0.0, 0.0, 0.0), (gx, 0.0, 0.0), step))
        return path if clear(path, costmap, margin) else None
    lead = clear_run(start, gx, costmap, margin)                  # straight on from the start
    tail = clear_run(goal, gx, costmap, margin, backward=True)    # straight in to the goal
    best, best_t = None, float('inf')
    for length in S_LENGTHS:
        if length > gx or not s_fits_turning_circles(length, gy, shapes):
            continue
        s = quintic_s(float(length), gy, step)
        xs = 0.0
        while xs + length <= gx + 1e-9:
            if xs > lead + 1e-9:
                break                    # the straight before the S would hit something
            if gx - (xs + length) <= tail + 1e-9:
                middle = to_world(start, [(x + xs, y, th, g) for x, y, th, g in s])
                if clear(middle, costmap, margin):
                    local = (straight((0.0, 0.0, 0.0), (xs, 0.0, 0.0), step) if xs > 1e-6 else [])
                    local += [(x + xs, y, th, g) for x, y, th, g in s]
                    if gx - (xs + length) > 1e-6:
                        local += straight((xs + length, gy, 0.0), (gx, gy, 0.0), step)
                    path = to_world(start, local)
                    t = drive_time(path, shapes)
                    if t < best_t:
                        best, best_t = path, t
                    break                # the earliest start that fits, for this length
            xs += S_EARLIEST_STEP
    return best


# ------------------------------------------------------------- arc S ----

CURVE_SCALES = (6.0, 4.0, 3.0, 2.0, 1.5, 1.0)   # x the smallest planned turning circle


def arc_s_path(start: Pose, goal: Pose, costmap, shapes: Shapes, margin: float = 0.04,
               step: float = 0.02) -> Optional[List[DensePose]]:
    """start -> goal (same heading) as the FASTEST of these shapes that fits:
    straight, a curve out, a straight diagonal, a curve back - tried with curves
    from wide down to the smallest planned turning circle, each turning out as
    early as it fits. None if nothing fits."""
    gx, gy, _ = to_local(start, goal)
    side = 1.0 if gy > 0 else -1.0
    best, best_t = None, float('inf')
    for scale in CURVE_SCALES:
        r1, r2 = ((shapes.r_left * scale, shapes.r_right * scale) if side > 0
                  else (shapes.r_right * scale, shapes.r_left * scale))
        xs = 0.0
        while xs < gx:
            local = _arc_s(xs, gx - xs, abs(gy), side, r1, r2, step)
            if local:
                path = to_world(start, local)
                if clear(path, costmap, margin):
                    t = drive_time(path, shapes)
                    if t < best_t:
                        best, best_t = path, t
                    break           # the earliest turn that fits, for this curve size
            xs += 0.05
    return best


def _arc_s(xs, d, h, side, r1, r2, step):
    """Straight 0 -> xs, arc r1 up to angle a, straight L, arc r2 back to 0,
    ending d ahead of xs and h to the side. None if it can't be done."""
    R = r1 + r2
    amax = math.asin(min(1.0, d / R)) if d < R else math.pi / 2.0 - 1e-6

    def lateral(a):     # lateral offset reached with turn angle a (L from the forward distance)
        L = (d - R * math.sin(a)) / math.cos(a)
        return R * (1 - math.cos(a)) + L * math.sin(a), L
    if lateral(amax)[0] < h:
        return None
    lo, hi = 1e-6, amax
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if lateral(mid)[0] < h else (lo, mid)
    a = (lo + hi) / 2.0
    L = lateral(a)[1]
    pts = [(t, 0.0, 0.0, 1) for t in np.arange(0.0, xs, step)]
    for t in np.arange(0.0, a, step / r1):                                   # turn out
        pts.append((xs + r1 * math.sin(t), side * r1 * (1 - math.cos(t)), side * t, 1))
    px, py = xs + r1 * math.sin(a), side * r1 * (1 - math.cos(a))
    for t in np.arange(0.0, L, step):                                        # the diagonal
        pts.append((px + t * math.cos(a), py + side * t * math.sin(a), side * a, 1))
    px, py = px + L * math.cos(a), py + side * L * math.sin(a)
    cx, cy = px + r2 * math.sin(a), py - side * r2 * math.cos(a)
    for u in np.arange(0.0, a, step / r2):                                   # turn back
        pts.append((cx - r2 * math.sin(a - u), cy + side * r2 * math.cos(a - u), side * (a - u), 1))
    pts.append((cx, cy + side * r2, 0.0, 1))                                 # exactly the goal
    return [(float(x), float(y), float(th), g) for x, y, th, g in pts]


# ------------------------------------------------------------- legs ----

def s_path(start: Pose, goal: Pose, costmap, shapes: Shapes, margin: float = 0.04,
           step: float = 0.02, shape: str = 'any') -> Optional[List[DensePose]]:
    """shape 'smooth' / 'arc': that S; 'any': the smooth S if one fits, else the arc S."""
    path = smooth_s_path(start, goal, costmap, shapes, margin, step) if shape in ('smooth', 'any') else None
    if path is None and shape in ('arc', 'any'):
        path = arc_s_path(start, goal, costmap, shapes, margin, step)
    return path


def lined_up_path(start: Pose, goal: Pose, costmap, shapes: Shapes, straight_m: float,
                  margin: float = 0.04, shape: str = 'any') -> Optional[List[DensePose]]:
    """start -> goal (same heading) as an S that is straight again `straight_m`
    before the goal, then straight in: the car is parallel to whatever is beside
    the goal when it gets there. None if it doesn't fit."""
    entry = (goal[0] - straight_m * math.cos(goal[2]), goal[1] - straight_m * math.sin(goal[2]), goal[2])
    path = s_path(start, entry, costmap, shapes, margin, shape=shape)
    if not path:
        return None
    if straight_m < 1e-6:
        return path
    tail = straight(entry, goal)
    if not clear(tail, costmap, margin):
        return None
    return path + tail


LADDER = ((1.0, 1.0), (0.5, 1.0), (0.5, 0.5), (0.0, 0.5))   # (x straight_m, x margin), tried in turn


def stepped_lined_up_path(start: Pose, goal: Pose, costmap, shapes: Shapes, straight_m: float,
                          margin: float) -> Optional[List[DensePose]]:
    """lined_up_path(), with less straight-in and less spare room in tight
    layouts before giving up on clean geometry (Hybrid A* makes the car wait).
    Every smooth S is tried before any arc S: the smooth S ends pointing straight
    already, the arcs need the straight-in to settle."""
    for shape in ('smooth', 'arc'):
        for f_straight, f_margin in LADDER:
            path = lined_up_path(start, goal, costmap, shapes, straight_m * f_straight, margin * f_margin, shape)
            if path:
                return path
    return None


U_TRANSITIONS = (0.25, 0.15, 0.0)   # m over which the U-turn's curvature ramps in/out (0.40: ~0.3 s slower)


def u_turn(length_ramp: float, k: float, turn: float, step: float = 0.005):
    """A U-turn from the origin facing +x: curvature ramping 0 -> k over length_ramp,
    k until the heading has turned pi, ramping back to 0 (turn +1 left, -1 right).
    Symmetric, so it ends level with its start (x = 0), facing back. (x, y, theta) every step."""
    arc = math.pi / k - length_ramp                    # the constant part
    total = 2 * length_ramp + arc
    n = max(2, int(total / step))
    pts, x, y, th = [(0.0, 0.0, 0.0)], 0.0, 0.0, 0.0
    ds = total / n
    for i in range(n):
        sm = (i + 0.5) * ds
        kk = k * min(1.0, sm / length_ramp, (total - sm) / length_ramp) if length_ramp > 0 else k
        th_mid = th + turn * kk * ds / 2.0
        x, y = x + ds * math.cos(th_mid), y + ds * math.sin(th_mid)
        th += turn * kk * ds
        pts.append((x, y, th))
    return pts


def round_path(a: Pose, b: Pose, costmap, shapes: Shapes, step: float = 0.02) -> Optional[List[DensePose]]:
    """a -> b round the back of obstacle 2 (they face opposite ways, level with the
    bar's centre): a U-turn whose curvature ramps in and out (U_TRANSITIONS, the
    longest that fits) - the half circle it replaces jumped from straight to full
    curvature, and the car ran ~9 cm wide toward the side walls (sim 2026-10-14).
    The plain half circle is the last resort. None if nothing fits."""
    d = abs(b[1] - a[1])
    turn = 1.0 if (b[1] - a[1]) * math.cos(a[2]) > 0 else -1.0     # left (counter-clockwise) or right
    k_limit = 1.0 / (shapes.r_left if turn > 0 else shapes.r_right)
    for ramp in U_TRANSITIONS:
        k_hi = min(k_limit, math.pi / ramp) if ramp > 0 else k_limit
        k_lo = 1e-3 if ramp == 0 else math.pi / (ramp + 50.0)
        # Wider across for a gentler curve: find the k that crosses exactly d.
        if abs(u_turn(ramp, k_hi, turn, 0.01)[-1][1]) > d:
            continue                     # even the sharpest allowed is too wide: shorter ramp
        if ramp == 0:
            k = 2.0 / d
        else:
            lo, hi = k_lo, k_hi
            for _ in range(40):
                mid = (lo + hi) / 2.0
                lo, hi = (mid, hi) if abs(u_turn(ramp, mid, turn, 0.01)[-1][1]) > d else (lo, mid)
            k = (lo + hi) / 2.0
        if k > k_limit + 1e-9:
            continue
        local = u_turn(ramp, k, turn, 0.005)
        # Resample every `step`, end exactly at b.
        out, acc = [], 0.0
        for (x0, y0, _), (x1, y1, t1) in zip(local, local[1:]):
            acc += math.hypot(x1 - x0, y1 - y0)
            if acc >= step:
                out.append((x1, y1, t1, 1))
                acc = 0.0
        path = to_world(a, out[:-1]) + [(b[0], b[1], b[2], 1)]
        if clear(path, costmap):
            return path
    return None


def has_reverse(path: Optional[List[DensePose]]) -> bool:
    return bool(path) and any(p[3] < 0 for p in path)
