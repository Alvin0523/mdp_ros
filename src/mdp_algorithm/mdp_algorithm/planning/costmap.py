#!/usr/bin/env python3
"""
The arena, its obstacles, the car's footprint and the Nav2-style costmap - the
one map the planner, the checkpoint check and Foxglove's /occupancy_grid use.

ARENA: the 2 x 2 m table, 20 x 20 cells of 10 cm (the tablet's grid). An
obstacle is a 10 x 10 cm block filling one cell; its image faces N/E/S/W.

FOOTPRINT: a rectangle around base_link (the rear-axle centre) - robot.footprint_*
in mdp_bringup/config/navigation.yaml - grown on every side by
costmap.footprint_padding (Nav2 `footprint_padding`).

COSTMAP: built like a Nav2 costmap with a static layer plus an inflation layer:
  * LETHAL (254): cells inside an obstacle block. Everything outside the arena
    counts as lethal too (off the map = collision, and the edge is inflated
    like a wall).
  * INSCRIBED (253): within the footprint's inscribed radius of a lethal cell -
    base_link cannot be there without the body touching it.
  * 252 * exp(-cost_scaling_factor * (d - inscribed)) out to inflation_radius,
    d = distance from the cell to the nearest lethal cell; 0 beyond.

COLLISION and COST of a pose follow Nav2's GridCollisionChecker for a polygon
footprint: look up the cost at base_link; if it is below the cost at the
circumscribed radius, the body cannot reach anything lethal and that cost is
the answer. Otherwise rasterise the padded footprint; the pose collides if any
of its cells is LETHAL (or off the map), and its cost is the highest of them.
Beyond Nav2, which checks only the outline, points inside the footprint are
checked too: a 10 cm block fits inside the 20 cm wide car, and a pose placed
straight onto one (a checkpoint, a shot's first sample) would pass an
outline-only check. When inflation_radius is smaller than the circumscribed
radius the shortcut is not safe, so the footprint is always checked (Nav2 warns
about this for speed; the result is still correct).

Settings are read when a Costmap is built (utils/params.py). Units here:
centimetres and radians - the planner's internal unit; the settings are in
metres. Cost values 0..254 as in nav2_costmap_2d.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from ..utils import params as planner_params

ARENA_SIZE_CM = 200.0     # the 2 x 2 m table
CELL_SIZE_CM = 10.0       # the tablet's grid cell
OBSTACLE_SIZE_CM = 10.0   # an obstacle block fills one cell

FREE = 0
MAX_NON_OBSTACLE = 252
INSCRIBED = 253
LETHAL = 254


@dataclass
class Obstacle:
    """An obstacle block. x_cm/y_cm: the CENTRE of the 10 x 10 cm block, arena
    frame. facing: 'N'/'E'/'S'/'W' - the side carrying the image, i.e. the
    direction the car must look from to see it. id: index in the caller's list."""
    x_cm: float
    y_cm: float
    facing: str
    id: int = -1


def snap_to_cell_centre(v_cm: float) -> float:
    """Centre of the 10 cm cell containing v_cm (cm). Obstacles and checkpoints live
    on cells, never on grid lines: cell 10 is 100..110 cm, centre 105."""
    return float(np.floor(v_cm / CELL_SIZE_CM + 1e-9) * CELL_SIZE_CM + CELL_SIZE_CM / 2.0)


class Footprint:
    """The padded footprint for one set of settings (default: the active ones)."""

    def __init__(self, params: planner_params.PlannerParams = None):
        p = params or planner_params.ACTIVE
        pad = p.footprint_padding * 100.0
        self.rear = p.footprint_rear * 100.0 + pad
        self.front = p.footprint_front * 100.0 + pad
        self.half_w = p.footprint_half_width * 100.0 + pad
        # Padded polygon, base_link frame, counter-clockwise.
        self.polygon = ((-self.rear, -self.half_w), (self.front, -self.half_w),
                        (self.front, self.half_w), (-self.rear, self.half_w))
        # Nav2 names: the largest circle about base_link that fits inside the
        # polygon, and the smallest one that contains it.
        self.inscribed_radius = min(self.rear, self.front, self.half_w)
        self.circumscribed_radius = max(math.hypot(px, py) for px, py in self.polygon)

    def check_points(self, spacing_cm: float, interior_spacing_cm: float = 2.5):
        """Points along the polygon's edges, at most `spacing_cm` apart, plus a
        grid of points inside it every `interior_spacing_cm` (2.5 cm cannot
        straddle a 10 cm block), as two arrays (x, y) in the base_link frame."""
        poly = self.polygon
        xs, ys = [], []
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
            n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing_cm)))
            t = np.arange(n) / n
            xs.append(x0 + (x1 - x0) * t)
            ys.append(y0 + (y1 - y0) * t)
        gx, gy = np.meshgrid(np.arange(-self.rear, self.front, interior_spacing_cm),
                             np.arange(-self.half_w, self.half_w, interior_spacing_cm))
        xs.append(gx.ravel())
        ys.append(gy.ravel())
        return np.concatenate(xs), np.concatenate(ys)


class Costmap:
    """The arena costmap for a set of obstacles. `obstacles` is kept as given."""

    def __init__(self, obstacles: List[Obstacle], params: planner_params.PlannerParams = None,
                 resolution_cm: float = None):
        p = params or planner_params.ACTIVE
        self.obstacles = list(obstacles)
        assert len(self.obstacles) <= 8   # the arena has at most 8 obstacles
        self.footprint = Footprint(p)
        self.inflation_radius = p.inflation_radius * 100.0      # cm
        self.cost_scaling = p.cost_scaling_factor / 100.0        # per cm (Nav2's is per metre)
        self.resolution = resolution_cm if resolution_cm is not None else p.resolution * 100.0
        self.size = int(round(ARENA_SIZE_CM / self.resolution))
        c = (np.arange(self.size) + 0.5) * self.resolution
        X, Y = np.meshgrid(c, c, indexing='ij')                  # [i = x][j = y]
        # Distance from each cell centre to the nearest lethal thing: the arena
        # edge (lethal beyond it) or a block's 10 x 10 cm square (0 inside it).
        half = OBSTACLE_SIZE_CM / 2.0
        d = np.minimum(np.minimum(X, ARENA_SIZE_CM - X), np.minimum(Y, ARENA_SIZE_CM - Y))
        for o in self.obstacles:
            d = np.minimum(d, np.hypot(np.maximum(np.abs(X - o.x_cm) - half, 0.0),
                                       np.maximum(np.abs(Y - o.y_cm) - half, 0.0)))
        self.distance = d
        self.cost = self.inflation_cost(d)
        self._px, self._py = self.footprint.check_points(self.resolution)
        # Nav2 findCircumscribedCost(): below this at base_link, the body cannot
        # reach a lethal cell. -1 = no shortcut (inflation radius too small).
        if self.inflation_radius >= self.footprint.circumscribed_radius:
            self.possible_collision_cost = int(self.inflation_cost(self.footprint.circumscribed_radius))
        else:
            self.possible_collision_cost = -1

    def inflation_cost(self, d_cm):
        """nav2 InflationLayer::computeCost for a distance to the nearest lethal
        cell (cm; scalar or array)."""
        d = np.asarray(d_cm, dtype=float)
        ins = self.footprint.inscribed_radius
        cost = np.where(d <= self.inflation_radius,
                        np.floor(MAX_NON_OBSTACLE * np.exp(-self.cost_scaling * (d - ins))), FREE)
        cost = np.where(d <= ins, INSCRIBED, cost)
        return np.where(d <= 0.0, LETHAL, cost).astype(np.uint8)

    def cost_at(self, x: float, y: float) -> int:
        i, j = int(x // self.resolution), int(y // self.resolution)
        if not (0 <= i < self.size and 0 <= j < self.size):
            return LETHAL
        return int(self.cost[i, j])

    def footprint_cost(self, x: float, y: float, theta: float) -> int:
        """Cost of the car at base_link pose (x, y, theta): LETHAL if it collides."""
        centre = self.cost_at(x, y)
        if centre == LETHAL:
            return LETHAL
        if 0 <= self.possible_collision_cost and centre < self.possible_collision_cost:
            return centre
        c, s = math.cos(theta), math.sin(theta)
        i = np.floor((x + c * self._px - s * self._py) / self.resolution).astype(int)
        j = np.floor((y + s * self._px + c * self._py) / self.resolution).astype(int)
        if i.min() < 0 or j.min() < 0 or i.max() >= self.size or j.max() >= self.size:
            return LETHAL
        return max(centre, int(self.cost[i, j].max()))

    def in_collision(self, x: float, y: float, theta: float) -> bool:
        return self.footprint_cost(x, y, theta) >= LETHAL

    def occupancy_grid_data(self):
        """Row-major (y outer) nav_msgs/OccupancyGrid data with Nav2's cost
        translation: 0 free, 1..98 inflation, 99 inscribed, 100 lethal."""
        t = np.zeros(256, dtype=np.int8)
        t[1:253] = 1 + (97 * (np.arange(1, 253) - 1)) // 251
        t[INSCRIBED] = 99
        t[LETHAL] = 100
        t[255] = -1
        return t[self.cost.T].ravel().tolist()
