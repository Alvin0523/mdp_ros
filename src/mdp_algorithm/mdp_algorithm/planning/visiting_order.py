#!/usr/bin/env python3
"""
Where to stop for each obstacle (its checkpoint) and in which order to visit
them. Ported from the teammate's mdp_algo package (pathfinding/hamiltonian.py),
reduced on 2026-09-27 to what the planner uses.

CHECKPOINT: one per obstacle, straight out from its image face:
    position = obstacle centre + planner.checkpoint_standoff * facing direction
    heading  = facing back at the obstacle, turned by theta_offset for the
               camera's mounting angle (task 1: the camera looks LEFT)
snapped to the centre of its 10 cm cell. Where the car does not fit there -
its nose (or tail) over the table's edge, e.g. block (1,18) facing E, or on
another block, e.g. (13,11)E with (16,13)W - the stop slides along the car's
heading (along the face), 1 cm at a time up to MAX_SLIDE_CM either way, to the
nearest spot where it fits and no other block is within the stand-off. An
obstacle with no such spot is unreachable and skipped.

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


def _blocked_by_any_obstacle(costmap: Costmap, x: float, y: float) -> bool:
    """Is (x, y) closer than the stand-off to ANY obstacle's centre? True
    Euclidean distance, not a grid lookup: a checkpoint sits exactly the
    stand-off from its own obstacle (so is not blocked by it), and a grid cell
    snap once made such points look closer than they are (2026-09-05)."""
    standoff = planner_params.ACTIVE.checkpoint_standoff_cm
    return any(np.hypot(x - o.x_cm, y - o.y_cm) < standoff for o in costmap.obstacles)


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

    # The car must fit there: on the table and clear of every block, and no other
    # block within the stand-off. Else the nearest spot along the face that is
    # (2026-10-07: the table's edge, then a block beside the stop).
    c, s = np.cos(theta), np.sin(theta)
    for d in [0.0] + [k * sign for k in np.arange(1.0, MAX_SLIDE_CM + 0.5) for sign in (-1.0, 1.0)]:
        sx, sy = x + d * c, y + d * s
        if not _blocked_by_any_obstacle(costmap, sx, sy) and not costmap.in_collision(sx, sy, theta):
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
