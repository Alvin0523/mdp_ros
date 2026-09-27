#!/usr/bin/env python3
"""
Where to stop for each obstacle (its checkpoint) and in which order to visit
them. Ported from the teammate's mdp_algo package (pathfinding/hamiltonian.py),
reduced on 2026-09-27 to what the planner uses.

CHECKPOINT: one per obstacle, straight out from its image face:
    position = obstacle centre + planner.checkpoint_standoff * facing direction
    heading  = facing back at the obstacle, turned by theta_offset for the
               camera's mounting angle (task 1: the camera looks LEFT)
snapped to the centre of its 10 cm cell. An obstacle whose checkpoint lies
too close to another obstacle, or where the car's footprint would not fit, is
unreachable and skipped.

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

    if _blocked_by_any_obstacle(costmap, x, y):
        return None
    # The car must fit there: on the table and clear of every block. This is
    # what makes an obstacle facing off the table, or one boxed in by another,
    # unreachable instead of planned to an impossible pose.
    if costmap.in_collision(x, y, theta):
        return None
    return (x, y, theta, obstacle.id)


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
