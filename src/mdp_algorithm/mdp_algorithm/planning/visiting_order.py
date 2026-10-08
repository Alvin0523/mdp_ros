#!/usr/bin/env python3
"""
Where to stop for each obstacle (its checkpoint) and in which order to visit
them. Ported from the teammate's mdp_algo package (pathfinding/hamiltonian.py),
reduced on 2026-09-27 to what the planner uses.

CHECKPOINT: one per obstacle, straight out from its image face:
    position = obstacle centre + planner.checkpoint_standoff * facing direction
    heading  = facing back at the obstacle, turned by theta_offset for the
               camera's mounting angle (task 1: the camera looks LEFT)
snapped to the centre of its 10 cm cell. Where the car does not fit there - its
nose (or tail) over the table's edge, e.g. block (1,18) facing E, or on another
block, e.g. (13,11)E with (16,13)W - or another block stands between it and the
face, the stop slides along the car's heading (along the face), 1 cm at a time up
to MAX_SLIDE_CM either way, to the nearest spot where it fits and sees the face.
An obstacle with no such spot is unreachable and skipped.

ORDER: every permutation of the reachable obstacles, scored by the sum of
Reeds-Shepp path lengths between consecutive checkpoints (exact for the <= 8
obstacles of the arena: at most 8! = 40320 orders, legs precomputed once).
Reeds-Shepp ignores obstacles, so this is the cheap estimate the runner can
use immediately; the legs themselves are then planned by hybrid_astar.py.

Units: centimetres and radians.
"""

from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np

from ..utils import geometry as utils
from ..utils import params as planner_params
from . import reeds_shepp as rs
from .costmap import Costmap, Obstacle, snap_to_cell_centre

Checkpoint = Tuple[float, float, float, int]  # (x_cm, y_cm, theta_rad, obstacle_id)

MAX_SLIDE_CM = 10.0   # a stop slid further than this: the camera misses the block
MAX_STANDOFF_SHIFT_CM = 3.0   # a stop may also be this much nearer the face or further from it


def _view_blocked(costmap: Costmap, obstacle: Obstacle, x: float, y: float) -> bool:
    """Does another block stand between the stop (x, y) and `obstacle`'s image
    face? The straight line from the stop to the face's centre, against every
    other block's square. (Was: any block's centre within the stand-off of the
    stop - that also threw away stops with a block merely beside them, e.g.
    (16,12)W with (11,12) 20 cm to the side, 2026-10-09.)"""
    a = utils.facing_to_rad(obstacle.facing)
    fx, fy = np.cos(a), np.sin(a)
    face = (obstacle.x_cm + fx * obstacle.size_x_cm / 2.0, obstacle.y_cm + fy * obstacle.size_y_cm / 2.0)
    for o in costmap.obstacles:
        if o is obstacle:
            continue
        x0, x1 = o.x_cm - o.size_x_cm / 2.0, o.x_cm + o.size_x_cm / 2.0
        y0, y1 = o.y_cm - o.size_y_cm / 2.0, o.y_cm + o.size_y_cm / 2.0
        for k in range(41):
            t = k / 40.0
            px, py = x + (face[0] - x) * t, y + (face[1] - y) * t
            if x0 <= px <= x1 and y0 <= py <= y1:
                return True
    return False


def obstacle_to_checkpoint(costmap: Costmap, obstacle: Obstacle,
                           theta_offset: float) -> Optional[Checkpoint]:
    """The obstacle's checkpoint (see module docstring), or None if unreachable."""
    facing_rad = utils.facing_to_rad(obstacle.facing)
    standoff = planner_params.ACTIVE.checkpoint_standoff_cm
    x = obstacle.x_cm + standoff * np.cos(facing_rad)
    y = obstacle.y_cm + standoff * np.sin(facing_rad)
    theta = utils.M(facing_rad + np.pi - theta_offset)

    # A checkpoint is a cell, like everything the tablet sees. For cell-centred
    # obstacles (the tablet's convention) this changes nothing.
    x, y = snap_to_cell_centre(x), snap_to_cell_centre(y)

    # The car must fit there (on the table, clear of every block, with the padding)
    # and see the face (no block in between). Else the nearest spot along the face
    # that does (2026-10-07: the table's edge, then a block beside the stop).
    # Also up to MAX_STANDOFF_SHIFT_CM nearer the face or further from it: a block
    # beside the stop can leave the car short of room by a cm (2026-10-09: (16,12)W
    # with (11,12) beside it, 0.5 cm). The smallest change wins; the IR fix takes
    # the gap the car really has, so a cm nearer or further does not matter to it.
    c, s = np.cos(theta), np.sin(theta)
    nx, ny = np.cos(facing_rad), np.sin(facing_rad)
    slides = [k for k in np.arange(-MAX_SLIDE_CM, MAX_SLIDE_CM + 0.5)]
    shifts = [k for k in np.arange(-MAX_STANDOFF_SHIFT_CM, MAX_STANDOFF_SHIFT_CM + 0.5)]
    for d, e in sorted(((d, e) for d in slides for e in shifts), key=lambda de: (abs(de[0]) + abs(de[1]), abs(de[1]))):
        sx, sy = x + d * c + e * nx, y + d * s + e * ny
        if not costmap.in_collision(sx, sy, theta) and not _view_blocked(costmap, obstacle, sx, sy):
            return (float(sx), float(sy), theta, obstacle.id)
    return None


def find_visiting_order(costmap: Costmap, start: Tuple[float, float, float],
                        theta_offset: float, turn_radius_cm: float,
                        ) -> Tuple[List[Obstacle], List[Checkpoint], List[Obstacle]]:
    """(obstacles in visiting order, their checkpoints, unreachable obstacles).
    start: (x_cm, y_cm, theta) of base_link; turn_radius_cm: the Reeds-Shepp
    radius used to score legs."""
    reachable, checkpoints, unreachable = [], [], []
    for obstacle in costmap.obstacles:
        cp = obstacle_to_checkpoint(costmap, obstacle, theta_offset)
        if cp is None:
            unreachable.append(obstacle)
        else:
            reachable.append(obstacle)
            checkpoints.append(cp)
    n = len(reachable)
    if n == 0:
        return [], [], unreachable

    # dist[i][j]: from positions[i] (0 = start, i = checkpoint i-1) to checkpoint j.
    positions = [start] + checkpoints
    dist = [[rs.get_optimal_path_length(positions[i], checkpoints[j], turn_radius_cm)
             if i != j + 1 else 0.0 for j in range(n)] for i in range(n + 1)]

    best, best_order = float('inf'), ()
    for order in permutations(range(n)):
        total = dist[0][order[0]]
        for k in range(n - 1):
            total += dist[order[k] + 1][order[k + 1]]
            if total >= best:
                break   # already worse than the best found
        if total < best:
            best, best_order = total, order
    return ([reachable[i] for i in best_order], [checkpoints[i] for i in best_order],
            unreachable)
