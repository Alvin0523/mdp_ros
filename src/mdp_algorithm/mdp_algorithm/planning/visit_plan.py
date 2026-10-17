#!/usr/bin/env python3
"""
Task 1 "anchors" visit plan (navigation.yaml visit.mode: anchors): fewer stops.

  ANCHOR     a square-on stop, as in square mode (visiting_order.py's checkpoint):
             the IR pair fixes the position there, the camera reads the image.
  DRIVE-BY   an image read while driving between two stops, at visit.drive_by_speed,
             no stop: the leg passes through a VIA point where the camera looks
             straight at it (visit.view_*), in one gear, never stopped at. One via
             per drive-by, so Hybrid A*'s real path sees it too.
  ANGLED     an image no square stop fits (blocks or the table's edge in the way):
             a stop further back and/or turned (up to 60 deg, 25-45 cm) where the
             camera still sees it - read there, no IR fix.
  LAST       the last image: an angled stop (up to visit.last_stop_max_angle off
             square) - nothing is driven after it, a position fix is no use there.

How (2026-10-17, worked out offline on the 2026-10-14 tablet layout with the user):
  1. every reachable image an anchor; order = shortest tour over the anchors;
  2. each round, try making each anchor a drive-by: keep the change that saves
     the most time; repeat until none saves time. At most
     visit.max_drive_bys_per_leg drive-bys between two stops (the position fixes
     stay close together);
  3. the last stop angled.
Everything here uses Reeds-Shepp curves (no obstacles, one radius) to estimate
the legs and what the camera sees along them - fast. The runner then plans the
real legs with Hybrid A* through the vias; a drive-by still not read is visited
at the end of the run.

Time model for choosing (an estimate, not the car's real times): the follower's
speed, GEAR_S per direction change, STOP_S per stop, IR_S more at an anchor.

Units here: centimetres and radians inside (the planner's), metres at plan_visits().
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..utils import geometry as utils
from ..utils import params as planner_params
from . import reeds_shepp as rs
from .costmap import Costmap
from .planner import CM_PER_M, ObstacleSpec, Pose, build_costmap
from .visiting_order import obstacle_to_checkpoint

GEAR_S = 1.0          # s per direction change (stop, change gear, go)
STOP_S = 1.5          # s per stop (stopping, the scan)
IR_S = 1.0            # s more at an anchor (the IR fix)
VIA_S = 0.0           # s charged per via point (see the tuning note at plan_visits)
SAMPLE_CM = 2.5       # spacing of the poses checked along an estimated leg
VIA_ANGLES_DEG = range(-30, 31, 10)    # via points tried: these angles off square ...
VIA_DIST_CM = (25.0, 32.0)             # ... at these camera -> image distances
VIA_NEAR_CM = 100.0                    # only legs starting or ending this near the image are bent
MIN_VIEW_CM = 15.0                     # nearer than this the image does not fit the frame
# A stop for an image with no square spot: up to 60 deg turned, 25-45 cm off (still in the
# camera's view, read from further back - the user's call 2026-10-17: visit it, don't skip it).
ANGLED_STOP_DEG = range(-60, 61, 5)
ANGLED_STOP_CM = (25.0, 28.0, 30.0, 32.0, 36.0, 40.0, 45.0)


@dataclass
class Visit:
    """One stop, in visiting order. pose: base_link (x, y, theta), metres.
    vias: points the leg INTO this stop passes through (metres), in order.
    drive_bys: images (obstacle indices) read on that leg."""
    obstacle: int
    pose: Pose
    kind: str                                  # 'anchor' (square) | 'angled' | 'last'
    vias: List[Pose] = field(default_factory=list)
    drive_bys: List[int] = field(default_factory=list)


class _Planner:
    def __init__(self, costmap: Costmap, start_cm, camera, speed):
        p = planner_params.ACTIVE
        self.costmap, self.start, self.camera, self.speed = costmap, start_cm, camera, speed
        self.radius = max(p.plan_turn_radius_left_cm, p.plan_turn_radius_right_cm)
        self.dmax, self.bmax, self.imax = p.view_max_dist * CM_PER_M, p.view_max_bearing, p.view_max_incidence
        self.min_view = p.drive_by_min_view * CM_PER_M
        self.max_per_leg = p.max_drive_bys_per_leg
        self.last_angle = p.last_stop_max_angle
        self.obstacles = costmap.obstacles
        self.anchor: Dict[int, Tuple[float, float, float]] = {}
        self.angled = set()            # images whose stop is angled (no square spot fits)
        for o in self.obstacles:
            cp = obstacle_to_checkpoint(costmap, o, camera[2])
            if cp is not None:
                self.anchor[o.id] = cp[:3]
            else:
                pose = self.angled_stop(o.id)
                if pose is not None:
                    self.anchor[o.id] = pose
                    self.angled.add(o.id)
        self._rs, self._vr, self._via = {}, {}, {}

    def angled_stop(self, i):
        """A stop for an image no square stop fits (blocks or the table edge in the way):
        further back and/or turned, as long as the camera sees it - the least turned and
        nearest 30 cm that fits (ANGLED_STOP_*). None if there is none."""
        fx, fy, a = self.face(i)
        cx_, cy_, cyaw = self.camera
        best = None
        for deg in ANGLED_STOP_DEG:
            for d in ANGLED_STOP_CM:
                score = abs(deg) / 10.0 + abs(d - 30.0) / 5.0
                if best is not None and score >= best[0]:
                    continue
                b = a + math.radians(deg)
                cx, cy = fx + d * math.cos(b), fy + d * math.sin(b)
                th = utils.M(math.atan2(fy - cy, fx - cx) - cyaw)
                x = cx - (cx_ * math.cos(th) - cy_ * math.sin(th))
                y = cy - (cx_ * math.sin(th) + cy_ * math.cos(th))
                if not self.costmap.in_collision(x, y, th) and not self.blocked(cx, cy, i):
                    best = (score, (x, y, th))
        return best[1] if best else None

    # -- estimates -------------------------------------------------------------
    def rs(self, a, b):
        """(time s, samples) of the Reeds-Shepp leg a -> b."""
        key = (a, b)
        if key not in self._rs:
            pts = rs.sample_path(a, b, self.radius, SAMPLE_CM)
            length = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip([(*a, 1)] + pts, pts))
            gears = sum(1 for p, q in zip(pts, pts[1:]) if p[3] != q[3])
            self._rs[key] = (length / CM_PER_M / self.speed + gears * GEAR_S, pts)
        return self._rs[key]

    def face(self, i):
        o = self.obstacles[i]
        a = utils.facing_to_rad(o.facing)
        return o.x_cm + math.cos(a) * o.size_x_cm / 2.0, o.y_cm + math.sin(a) * o.size_y_cm / 2.0, a

    def cam(self, pose):
        x, y, th = pose[:3]
        cx, cy, cyaw = self.camera
        return x + cx * math.cos(th) - cy * math.sin(th), y + cx * math.sin(th) + cy * math.cos(th), th + cyaw

    def blocked(self, x, y, i):
        fx, fy, _ = self.face(i)
        for o in self.obstacles:
            if o.id == i:
                continue
            x0, x1 = o.x_cm - o.size_x_cm / 2.0, o.x_cm + o.size_x_cm / 2.0
            y0, y1 = o.y_cm - o.size_y_cm / 2.0, o.y_cm + o.size_y_cm / 2.0
            for k in range(1, 40):
                t = k / 40.0
                px, py = x + (fx - x) * t, y + (fy - y) * t
                if x0 <= px <= x1 and y0 <= py <= y1:
                    return True
        return False

    def sees(self, pose, i) -> bool:
        cx, cy, cth = self.cam(pose)
        fx, fy, a = self.face(i)
        dx, dy = fx - cx, fy - cy
        d = math.hypot(dx, dy)
        if not MIN_VIEW_CM <= d <= self.dmax:
            return False
        if abs(utils.M(math.atan2(dy, dx) - cth)) > self.bmax:
            return False
        if math.acos(max(-1.0, min(1.0, -(dx * math.cos(a) + dy * math.sin(a)) / d))) > self.imax:
            return False
        return not self.blocked(cx, cy, i)

    def view_run(self, pts, i) -> float:
        """The longest stretch (cm) of the path with image i in view."""
        best = run = 0.0
        prev = None
        for q in pts:
            if self.sees(q, i):
                if prev is not None:
                    run += math.hypot(q[0] - prev[0], q[1] - prev[1])
                prev = q
                best = max(best, run)
            else:
                run, prev = 0.0, None
        return best

    def leg_view(self, a, b, i) -> float:
        """view_run() of the estimated leg a -> b (cached)."""
        key = (a, b, i)
        if key not in self._vr:
            self._vr[key] = self.view_run(self.rs(a, b)[1], i)
        return self._vr[key]

    def via_options(self, i):
        """Poses whose camera looks straight at image i, 25-35 cm off, up to 35 deg off square (cached)."""
        if i not in self._via:
            self._via[i] = self._via_options(i)
        return self._via[i]

    def _via_options(self, i):
        fx, fy, a = self.face(i)
        cx_, cy_, cyaw = self.camera
        out = []
        for deg in VIA_ANGLES_DEG:
            for d in VIA_DIST_CM:
                b = a + math.radians(deg)
                cx, cy = fx + d * math.cos(b), fy + d * math.sin(b)
                th = utils.M(math.atan2(fy - cy, fx - cx) - cyaw)
                x = cx - (cx_ * math.cos(th) - cy_ * math.sin(th))
                y = cy - (cx_ * math.sin(th) + cy_ * math.cos(th))
                if not self.costmap.in_collision(x, y, th) and not self.blocked(cx, cy, i):
                    out.append(((x, y, th), abs(deg)))
        return out

    # -- a plan for a set of anchors ------------------------------------------
    def tour(self, anchors):
        """Shortest order (Held-Karp on the estimated times) of the anchors."""
        ids = list(anchors)
        n = len(ids)
        if n == 0:
            return []
        poses = [self.anchor[i] for i in ids]
        best = {(1 << k, k): (self.rs(self.start, poses[k])[0], None) for k in range(n)}
        for mask in range(1, 1 << n):
            for k in range(n):
                if (mask, k) not in best:
                    continue
                c0 = best[(mask, k)][0]
                for j in range(n):
                    if mask >> j & 1:
                        continue
                    key, c = (mask | 1 << j, j), c0 + self.rs(poses[k], poses[j])[0]
                    if key not in best or c < best[key][0]:
                        best[key] = (c, k)
        full = (1 << n) - 1
        k = min(range(n), key=lambda j: best[(full, j)][0])
        order, mask = [], full
        while k is not None:
            order.append(ids[k])
            prev = best[(mask, k)][1]
            mask &= ~(1 << k)
            k = prev
        return order[::-1]

    def evaluate(self, anchors):
        """(estimated time, legs) for these anchors, every other reachable image a
        drive-by - or None if one cannot be seen. legs: per stop {'obstacle', 'pose',
        'from', 'vias', 'drive_bys' (the image of each via, same order), 't'} (cm)."""
        order = self.tour(anchors)
        legs, prev = [], self.start
        for i in order:
            pose = self.anchor[i]
            legs.append({'obstacle': i, 'pose': pose, 'vias': [], 'drive_bys': [],
                         'from': prev, 't': self.rs(prev, pose)[0]})
            prev = pose
        for i in sorted(set(self.anchor) - set(anchors)):
            best = self.best_via(legs, i)
            if best is None:
                return None
            _, leg, pos, via, t = best
            leg['vias'].insert(pos, via)
            leg['drive_bys'].insert(pos, i)
            leg['t'] = t
        total = sum(leg['t'] + STOP_S + IR_S for leg in legs)
        return total, legs

    def chain(self, points):
        """Estimated time through `points`, or None if it turns round at one of them
        (a via is driven through in one gear)."""
        t, last_gear = 0.0, None
        for k, (a, b) in enumerate(zip(points, points[1:])):
            tk, pts = self.rs(a, b)
            if not pts:
                return None
            if k > 0 and pts[0][3] != last_gear:
                return None
            t, last_gear = t + tk, pts[-1][3]
        return t

    def via_seen(self, points, k, i) -> bool:
        """Image i in view for drive_by_min_view around via points[k]."""
        return max(self.leg_view(points[k - 1], points[k], i), self.leg_view(points[k], points[k + 1], i)) >= self.min_view

    def best_via(self, legs, i):
        """The cheapest way to read image i driving by: a VIA point (the camera looking
        straight at it, via_options()) put into one leg - (extra s, leg, position,
        via, new leg time), or None. Every drive-by gets its own via: Hybrid A* must pass
        through it, so the real path sees the image too (a plain leg that only the
        Reeds-Shepp estimate saw it from did not - sim 2026-10-17)."""
        fx, fy, _ = self.face(i)
        cands = []
        for leg in legs:
            if len(leg['vias']) >= self.max_per_leg:
                continue
            pts = [leg['from']] + leg['vias'] + [leg['pose']]
            if min(math.hypot(p[0] - fx, p[1] - fy) for p in pts) > VIA_NEAR_CM:
                continue
            for via, _ in self.via_options(i):
                for pos in range(len(leg['vias']) + 1):
                    a, b = pts[pos], pts[pos + 1]
                    # a lower bound on the extra time: a curve is never shorter than the straight line
                    lb = (math.hypot(via[0] - a[0], via[1] - a[1]) + math.hypot(b[0] - via[0], b[1] - via[1])
                          ) / CM_PER_M / self.speed - self.rs(a, b)[0]
                    cands.append((lb, id(leg), leg, pos, via))
        cands.sort(key=lambda c: c[0])
        best = None
        for lb, _, leg, pos, via in cands:
            if best is not None and lb >= best[0]:
                break                                    # nothing further can beat it
            pts = [leg['from']] + leg['vias'][:pos] + [via] + leg['vias'][pos:] + [leg['pose']]
            t = self.chain(pts)
            if t is not None:
                t += VIA_S * (len(leg['vias']) + 1)      # Hybrid A* bends round a via more than Reeds-Shepp
            if t is None or (best is not None and t - leg['t'] >= best[0]):
                continue
            imgs = leg['drive_bys'][:pos] + [i] + leg['drive_bys'][pos:]
            if all(self.via_seen(pts, k + 1, j) for k, j in enumerate(imgs)):
                best = (t - leg['t'], leg, pos, via, t)
        return best

    def last_stop(self, legs):
        """Make the last stop angled when that is quicker and keeps its leg's drive-bys."""
        leg = legs[-1]
        frm = leg['vias'][-1] if leg['vias'] else leg['from']
        best = (self.rs(frm, leg['pose'])[0], leg['pose'])
        for pose, deg in self.via_options(leg['obstacle']):
            if math.radians(deg) > self.last_angle:
                continue
            t, pts = self.rs(frm, pose)
            if t >= best[0]:
                continue
            points = [leg['from']] + leg['vias'] + [pose]
            if leg['vias'] and self.chain(points[-3:]) is None:
                continue                                 # would turn round at the last via
            if all(self.via_seen(points, k + 1, j) for k, j in enumerate(leg['drive_bys'])):
                best = (t, pose)
        leg['pose'] = best[1]


def plan_visits(obstacles_grid: List[ObstacleSpec], start_pose_m: Pose, camera_m: Tuple[float, float, float],
                ) -> Tuple[List[Visit], List[int], Costmap]:
    """The anchors-mode plan (module docstring).

    obstacles_grid: (x_cm, y_cm, facing) per obstacle, as plan_visiting_order().
    start_pose_m: base_link start pose, metres / radians.
    camera_m: the camera on base_link (x, y, yaw), metres / radians (URDF; task 1:
        (0, -0.055, +pi/2) - beside the rear axle, looking left).

    Returns (visits, unreachable, costmap): visits in order; unreachable obstacle
    indices (no checkpoint, skipped - as plan_visiting_order); the costmap.
    """
    costmap = build_costmap(obstacles_grid)
    start = (start_pose_m[0] * CM_PER_M, start_pose_m[1] * CM_PER_M, start_pose_m[2])
    camera = (camera_m[0] * CM_PER_M, camera_m[1] * CM_PER_M, camera_m[2])
    speed = min(0.5, max(0.05, planner_params.ACTIVE.desired_linear_vel))
    pl = _Planner(costmap, start, camera, speed)
    unreachable = [o.id for o in costmap.obstacles if o.id not in pl.anchor]
    anchors = sorted(pl.anchor)
    cur = pl.evaluate(anchors) if anchors else (0.0, [])
    while len(anchors) > 1:
        tried = [(pl.evaluate([a for a in anchors if a != d]), d) for d in anchors]
        tried = [(r, d) for r, d in tried if r is not None and r[0] < cur[0]]
        if not tried:
            break
        cur, drop = min(tried, key=lambda rd: rd[0][0])
        anchors = [a for a in anchors if a != drop]
    legs = cur[1]
    if legs:
        pl.last_stop(legs)

    def m(p):
        return (p[0] / CM_PER_M, p[1] / CM_PER_M, p[2])
    visits = [Visit(obstacle=leg['obstacle'], pose=m(leg['pose']),
                    kind='last' if k == len(legs) - 1 else 'angled' if leg['obstacle'] in pl.angled else 'anchor',
                    vias=[m(v) for v in leg['vias']], drive_bys=list(leg['drive_bys']))
              for k, leg in enumerate(legs)]
    return visits, unreachable, costmap
