#!/usr/bin/env python3
"""
Where to stop for each obstacle (its checkpoint) and in which order to visit
them. Ported from the teammate's mdp_algo package (pathfinding/hamiltonian.py),
reduced on 2026-09-27 to what the planner uses.

CHECKPOINT: one per obstacle, straight out from its image face:
    position = obstacle centre + planner.checkpoint_standoff * facing direction
    heading  = facing back at the obstacle, turned by theta_offset for the
               camera's mounting angle (task 1: the camera looks LEFT)
snapped to the centre of its 10 cm cell. Next to the table's edge the car's
nose (or tail) can stick out over it, e.g. block (1,18) facing E: the stop
then slides back along the car's heading, 1 cm at a time up to
MAX_EDGE_SLIDE_CM, until the car is on the table. An obstacle whose checkpoint
lies too close to another obstacle, or where the car does not fit even so, is
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

MAX_EDGE_SLIDE_CM = 10.0   # a stop slid further than this: the camera and IRs miss the block


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
    if not costmap.in_collision(x, y, theta):
        return (x, y, theta, obstacle.id)
    if not _over_edge(costmap, x, y, theta):
        return None
    # Over the table's edge: slide along the heading, away from it (2026-10-07).
    c, s = np.cos(theta), np.sin(theta)
    for d in np.arange(1.0, MAX_EDGE_SLIDE_CM + 0.5):
        for sx, sy in ((x - d * c, y - d * s), (x + d * c, y + d * s)):
            if (not _over_edge(costmap, sx, sy, theta) and not costmap.in_collision(sx, sy, theta)
                    and not _blocked_by_any_obstacle(costmap, sx, sy)):
                return (float(sx), float(sy), theta, obstacle.id)
    return None


def _over_edge(costmap: Costmap, x: float, y: float, theta: float) -> bool:
    """Does the car's padded footprint at (x, y, theta) stick out over the table's edge?"""
    f, c, s = costmap.footprint, np.cos(theta), np.sin(theta)
    for px in (f.front, -f.rear):
        for py in (f.half_w, -f.half_w):
            cx, cy = x + c * px - s * py, y + s * px + c * py
            if not (0.0 <= cx <= costmap.width_cm and 0.0 <= cy <= costmap.height_cm):
                return True
    return False


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
