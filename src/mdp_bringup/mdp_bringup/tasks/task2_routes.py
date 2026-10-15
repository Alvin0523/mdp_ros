"""Task 2's course costmap and the CP1 -> CP2 route table (ROS-free).

The leg checkpoint 1 -> checkpoint 2 must be ready the instant the car reaches
checkpoint 1: there it has just measured obstacle 2 (ultrasonic), arrow 2 (YOLO)
and the bar's width (a bullseye). So every such route is planned OFFLINE, on the
laptop, into config/task2_routes.json.gz, and the Pi only looks one up:

  d2         0.60 - 1.50 m, every 5 cm  (obstacle 1's far face -> the bar's near face)
  bar width  0.30 - 1.20 m, every 5 cm  (tasks.yaml task2 obstacle_2 width_range)
  arrows     LEFT/LEFT, LEFT/RIGHT, RIGHT/LEFT, RIGHT/RIGHT - all four: the car
             turns tighter left than right (navigation.yaml turning circles), so a
             route is not the mirror image of the other side's.

Each route is stored in checkpoint 1's frame (CP1 at the origin, facing +x). At
run time: the nearest d2 (the route's final straight stretched or shortened by
the <= 2.5 cm left) and the nearest bar width (the runner plans the rest of the
run with that width too). Nearest, not rounded up: the side walls stand a fixed
distance from the bar's REAL ends, so a width rounded up by up to 5 cm put the
assumed walls that much too far out (sim 2026-10-14: an 80 cm bar planned as 85
passed a wall at 3.8 cm), and placing the walls for the narrower bar instead left
81 of the routes needing a reverse. Nearest: the bar's end and the walls both
within 2.5 cm (+ the measurement's error), either way.

The table is only valid for the settings it was built with (fingerprint());
the runner checks and, on a mismatch, plans the leg live instead. Rebuild:
`pixi run task2-table` (tools/task2_table.py).
"""

import dataclasses
import gzip
import json
import math
import os
from typing import List, Optional, Tuple

import yaml

from mdp_algorithm.planning import task2_paths
from mdp_algorithm.planning.costmap import Costmap, Obstacle
from mdp_bringup.utils.obstacle_layout import TASK2_AREA_M, TASK2_CENTRE_Y, Task2Arena

SIDE = {'LEFT': 1.0, 'RIGHT': -1.0}       # +y = left
STEP_M = 0.05
D2_RANGE = (0.60, 1.50)                   # m, the course rules
TABLE_VERSION = 4       # how routes are built (shapes, walls): a table from another version is rebuilt
TABLE_FILE = 'task2_routes.json.gz'    # gzip: ~2.5 MB of JSON otherwise

# The settings a route depends on (navigation.yaml task2_runner): with the car's
# outline and turning circles, the course sizes and the planner's margin, the
# table's fingerprint.
ROUTE_SETTINGS = ('lane_offset', 'cp1_back', 'obstacle_1_padding', 'side_clearance', 'line_up_dist', 'obstacle_padding', 'wall_padding',
                  'straight_speed', 'max_lateral_accel')


def task2_settings(navigation_yaml: str) -> dict:
    """navigation.yaml's task2_runner parameters (the file, no ROS)."""
    with open(navigation_yaml) as f:
        return yaml.safe_load(f)['task2_runner']['ros__parameters']


def fingerprint(car, arena: Task2Arena, settings: dict) -> dict:
    """Everything a CP1 -> CP2 route depends on, rounded (a float printed twice
    must compare equal). car: PlannerParams."""
    values = {name: float(settings[name]) for name in ROUTE_SETTINGS}
    values['version'] = TABLE_VERSION
    values.update({
        'footprint': [car.footprint_rear, car.footprint_front, car.footprint_half_width],
        'turning': [car.minimum_turning_radius_left, car.minimum_turning_radius_right,
                    car.turning_radius_margin],
        'obstacle_1': list(arena.obstacle_1_size[:2]),
        'bar_depth': arena.obstacle_2_size[0],
        'side_wall_clearance': arena.side_wall_clearance,
    })
    return json.loads(json.dumps(values, default=float), parse_float=lambda s: round(float(s), 4))


def course_costmap(arena: Task2Arena, x1: float, x2: Optional[float], bar_width: float,
                   padding: float, wall_padding: Optional[float] = None,
                   obstacle_1_padding: Optional[float] = None) -> Costmap:
    """The course as far as it is known: carpark walls, the obstacles measured so
    far, and the side walls that MAY stand beside obstacle 2 (assumed there).
    padding: the car's margin round the obstacles (task 2's obstacle_padding, the
    planner's footprint_padding). wall_padding: round the walls (default the same):
    the costmap pads walls like obstacles, so they are drawn (padding - wall_padding)
    further back from the course - more room round the obstacles without squeezing
    the car toward the walls. obstacle_1_padding: more before and behind obstacle 1
    alone (drawn that much longer)."""
    cm = 100.0
    s1, s2 = arena.obstacle_1_size, arena.obstacle_2_size
    grow = 0.0 if obstacle_1_padding is None else 2.0 * max(0.0, obstacle_1_padding - padding)
    # Along the course only: the lane beside it keeps obstacle_padding (cp1_back 0.1-0: the
    # car beside it), while leg 1 crossing behind it keeps the more.
    obstacles = [Obstacle(x1 * cm, TASK2_CENTRE_Y * cm, 'W', 0, (s1[0] + grow) * cm, s1[1] * cm)]
    walls = arena.carpark_walls()
    if x2 is not None:
        obstacles.append(Obstacle(x2 * cm, TASK2_CENTRE_Y * cm, 'W', 1, s2[0] * cm, bar_width * cm))
        walls = walls + arena.side_walls(x2, bar_width)
    back = 0.0 if wall_padding is None else max(0.0, padding - wall_padding)
    walls = [_set_back(w, back) for w in walls]
    return Costmap(obstacles, arena_cm=(TASK2_AREA_M[0] * cm, TASK2_AREA_M[1] * cm),
                   walls=[tuple(v * cm for v in w) for w in walls], edge_padding_cm=padding * cm)


def arena_to_dict(arena: Task2Arena) -> dict:
    """The course rules for a JSON request (task2_runner -> the laptop's task1_planner)."""
    return dataclasses.asdict(arena)


def arena_from_dict(data: dict) -> Task2Arena:
    return Task2Arena(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def _set_back(wall, d: float):
    """A wall rectangle moved d away from the course: a side wall (or the carpark's)
    away from the centre line, the carpark's back wall back."""
    x0, y0, x1, y1 = wall
    if y0 >= TASK2_CENTRE_Y:
        return (x0, y0 + d, x1, y1 + d)
    if y1 <= TASK2_CENTRE_Y:
        return (x0, y0 - d, x1, y1 - d)
    return (x0 - d, y0, x1 - d, y1)            # across the centre line: the carpark's back wall


def checkpoint_1(x1: float, arrow_1: str, lane_offset: float, back: float = 0.0):
    """Beside obstacle 1 (centre x1), lane_offset out on arrow 1's side, `back`
    behind its centre (navigation.yaml cp1_back: the camera in front of it)."""
    return (x1 - back, TASK2_CENTRE_Y + SIDE[arrow_1] * lane_offset, 0.0)


def bar_checkpoints(x2: float, arrow_2: str, bar_width: float, side_clearance: float):
    """Checkpoints 2 (the bar's end on arrow 2's side) and 3 (its other end, heading back)."""
    s2, half = SIDE[arrow_2], bar_width / 2.0 + side_clearance
    return [(x2, TASK2_CENTRE_Y + s2 * half, 0.0), (x2, TASK2_CENTRE_Y - s2 * half, math.pi)]


def bar_x(arena: Task2Arena, x1: float, d2: float) -> float:
    """The bar's centre from obstacle 1's centre and d2 (face to face)."""
    return x1 + arena.obstacle_1_size[0] / 2.0 + d2 + arena.obstacle_2_size[0] / 2.0


def grid(lo: float, hi: float) -> List[float]:
    return [round(lo + STEP_M * k, 3) for k in range(int(round((hi - lo) / STEP_M)) + 1)]


def build_route(arena: Task2Arena, car, settings: dict, d2: float, bar_width: float, arrow_1: str,
                arrow_2: str, plan_leg=None, back: Optional[float] = None):
    """One CP1 -> CP2 route in CP1's frame, and how it was made ('smooth' / 'arc' /
    'hybrid_astar'), or (None, why). plan_leg: Hybrid A* (planner.plan_leg) for
    what no clean shape fits; None = clean shapes only."""
    shapes = task2_paths.Shapes.from_params(car, settings['straight_speed'], settings['max_lateral_accel'])
    x1 = arena.opening_x + D2_RANGE[0] + arena.obstacle_1_size[0] / 2.0     # d1 does not change this leg
    x2 = bar_x(arena, x1, d2)
    back = float(settings['cp1_back']) if back is None else back
    cp1 = checkpoint_1(x1, arrow_1, float(settings['lane_offset']), back)
    cp2 = bar_checkpoints(x2, arrow_2, bar_width, float(settings['side_clearance']))[0]
    costmap = course_costmap(arena, x1, x2, bar_width, float(settings['obstacle_padding']),
                             float(settings['wall_padding']), float(settings['obstacle_1_padding']))
    if costmap.in_collision(cp2[0] * 100.0, cp2[1] * 100.0, cp2[2]):
        return None, 'checkpoint 2 not clear'
    path, how = None, None
    for shape in ('smooth', 'arc'):
        for f_straight, f_margin in task2_paths.LADDER:
            path = task2_paths.lined_up_path(cp1, cp2, costmap, shapes, float(settings['line_up_dist']) * f_straight,
                                             0.04 * f_margin, shape)
            if path:
                how = shape
                break
        if path:
            break
    if not path and plan_leg is not None:
        path, how = plan_leg(costmap, cp1, cp2) or None, 'hybrid_astar'
    if not path:
        return None, 'no path'
    local = [task2_paths.to_local(cp1, p[:3]) + (int(p[3]),) for p in path]
    return local, how


def cp1_backs(settings: dict) -> List[float]:
    """Checkpoint 1's distances behind obstacle 1's centre the table covers: cp1_back,
    and 10 cm nearer at a time down to 0 - where a short course puts it (no room for
    the S that far back; task2_runner cp1_lined_up())."""
    back, out = float(settings['cp1_back']), []
    while back > -1e-6:
        out.append(round(max(0.0, back), 2))
        back -= 0.10
    return out


def route_key(arrow_1: str, arrow_2: str, back: float) -> str:
    return f'{arrow_1}/{arrow_2}@{round(back, 2):.2f}'


def build_table(arena: Task2Arena, car, settings: dict, plan_leg=None, progress=None) -> dict:
    """Every route (module docstring) for every cp1_backs(); progress(done, total, key, d2, width, how)."""
    lo, hi = arena.obstacle_2_width_range
    d2s, widths, backs = grid(*D2_RANGE), grid(lo, hi), cp1_backs(settings)
    table = {'fingerprint': fingerprint(car, arena, settings), 'd2': d2s, 'bar_width': widths, 'backs': backs,
             'routes': {}, 'how': {}}
    total, done = 4 * len(backs) * len(d2s) * len(widths), 0
    for back in backs:
        for a1 in ('LEFT', 'RIGHT'):
            for a2 in ('LEFT', 'RIGHT'):
                key = route_key(a1, a2, back)
                rows, hows = [], []
                for d2 in d2s:
                    row, how_row = [], []
                    for w in widths:
                        path, how = build_route(arena, car, settings, d2, w, a1, a2, plan_leg, back)
                        row.append(None if path is None else [[round(v, 4) for v in p[:3]] + [p[3]] for p in path])
                        how_row.append(how)
                        done += 1
                        if progress:
                            progress(done, total, key, d2, w, how)
                    rows.append(row)
                    hows.append(how_row)
                table['routes'][key], table['how'][key] = rows, hows
    return table


class RouteTable:
    """config/task2_routes.json.gz, loaded; lookup() at checkpoint 1."""

    def __init__(self, data: dict):
        self.data = data
        self.d2s, self.widths = data['d2'], data['bar_width']

    @classmethod
    def load(cls, path: str) -> Optional['RouteTable']:
        if not os.path.exists(path):
            return None
        with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
            return cls(json.load(f))

    def matches(self, fp: dict) -> bool:
        return self.data.get('fingerprint') == fp

    def round_width(self, width: float) -> Optional[float]:
        """The table's width nearest `width` (None: outside the table by more than a half step)."""
        w = min(self.widths, key=lambda v: abs(v - width))
        return w if abs(w - width) <= STEP_M / 2 + 1e-6 else None

    def lookup(self, d2: float, bar_width: float, arrow_1: str, arrow_2: str,
               back: float) -> Tuple[Optional[list], str]:
        """(route in CP1's frame ending exactly d2 ahead, why-not). bar_width must be
        one of the table's (round_width()); back: checkpoint 1 behind obstacle 1's
        centre (one of cp1_backs(), within 2 cm)."""
        backs = self.data.get('backs', [])
        near = min(backs, key=lambda b: abs(b - back)) if backs else None
        if near is None or abs(near - back) > 0.02:
            return None, f'checkpoint 1 {back * 100:.0f} cm back not in the table'
        # Up to 1.5 steps outside: the nearest route, its straight adjusted - the course's
        # d2 is 0.60-1.50 m, a reading a few cm outside is the ultrasonic / pose's error
        # (sim 2026-10-14: 0.57 m for 0.60, 1.55-1.56 m for 1.50).
        if not (self.d2s[0] - 1.5 * STEP_M <= d2 <= self.d2s[-1] + 1.5 * STEP_M):
            return None, f'd2 {d2:.2f} m outside the table'
        if bar_width not in self.widths:
            return None, f'bar width {bar_width:.2f} m not in the table'
        i = min(range(len(self.d2s)), key=lambda k: abs(self.d2s[k] - d2))
        route = self.data['routes'][route_key(arrow_1, arrow_2, near)][i][self.widths.index(bar_width)]
        if route is None:
            return None, 'no route for this layout (table)'
        return stretch(route, d2 - self.d2s[i]), ''


def stretch(route: list, dx: float) -> list:
    """The route with its final straight dx longer (shorter if < 0), so it ends dx
    further ahead. Without a straight (a Hybrid A* route), all of it shifts."""
    pts = [tuple(p) for p in route]
    if abs(dx) < 1e-6:
        return pts
    end = pts[-1]
    j = len(pts) - 1
    while j > 0 and abs(pts[j - 1][2] - end[2]) < 1e-3 and abs(pts[j - 1][1] - end[1]) < 1e-3:
        j -= 1
    x0 = pts[j - 1][0] if j > 0 else pts[0][0]
    span = end[0] - x0
    if span < 0.05:
        return pts[:j] + [(x + dx, y, th, g) for x, y, th, g in pts[j:]] if j else \
            [(x + dx * k / max(1, len(pts) - 1), y, th, g) for k, (x, y, th, g) in enumerate(pts)]
    return pts[:j] + [(x + dx * (x - x0) / span, y, th, g) for x, y, th, g in pts[j:]]
