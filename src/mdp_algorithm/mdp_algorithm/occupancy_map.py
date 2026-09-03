#!/usr/bin/env python3
"""
Collision-aware occupancy grid for the 200x200cm / 40x40-cell MDP arena.

Ported from the teammate's mdp_algo package (objects/OccupancyMap.py,
objects/Obstacle.py) - pygame (sprite rendering) and matplotlib
(interactive plotting) dependencies stripped, since this only needs to run
headless on the Pi as a planning input, never rendered. All internal units
are CENTIMETRES (matching the teammate's original grid math exactly, to
avoid introducing conversion bugs into ported search code) - callers at the
mdp_algorithm package boundary (e.g. collision_aware_planner.py) are
responsible for converting to/from metres before handing paths to
pure_pursuit_follower.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

GRID_SIZE = 40          # 40x40 cells
ARENA_SIZE_CM = 200.0   # 200x200cm arena
CELL_SIZE_CM = ARENA_SIZE_CM / GRID_SIZE   # 5cm/cell

# Obstacle/border inflation, in cells - kept at the teammate's tested value
# (3 cells = 15cm at this grid's 5cm/cell resolution, confirmed here, not
# assumed). Comfortably covers this robot's own footprint (~8cm half-width,
# traction_track_width/2, per docs/stm32/architecture.md) plus margin - no
# change needed for our chassis.
INFLATION_CELLS = 3


@dataclass
class Obstacle:
    """Grid-cell obstacle position + the direction its scannable image faces.

    x_g/y_g: bottom-left grid-cell coordinates (0..39), NOT continuous cm.
    facing: {'N', 'S', 'E', 'W'} - direction the agent must face to see the
    obstacle's image, matching the teammate's convention.
    """
    x_g: int
    y_g: int
    facing: str
    id: int = -1


def grid_to_coords(x_g: float, y_g: float):
    """Grid-cell coordinates -> continuous cm coordinates (cell origin)."""
    return x_g * CELL_SIZE_CM, y_g * CELL_SIZE_CM


def coords_to_grid(x: float, y: float):
    """Continuous cm coordinates -> grid-cell coordinates."""
    return int(x // CELL_SIZE_CM), int(y // CELL_SIZE_CM)


class OccupancyMap:
    def __init__(self, obstacles: List[Obstacle] = None) -> None:
        """
        Parameters:
            occupancy_grid (np.array): GRID_SIZE x GRID_SIZE binary grid,
                1 = occupied (border, inflated obstacle footprint), 0 = free.
        """
        obstacles = obstacles or []
        assert len(obstacles) <= 8   # arena has at most 8 obstacles

        self.xmin, self.xmax, self.ymin, self.ymax = 0.0, ARENA_SIZE_CM, 0.0, ARENA_SIZE_CM
        self.obstacles: List[Obstacle] = []
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE))

        self.add_obstacles_to_grid(obstacles)

    def add_obstacles_to_grid(self, obstacles: List[Obstacle]) -> None:
        assert len(self.obstacles) + len(obstacles) <= 8
        self.obstacles += obstacles

        n = INFLATION_CELLS
        self.occupancy_grid[:n, :] = 1
        self.occupancy_grid[-n:, :] = 1
        self.occupancy_grid[:, :n] = 1
        self.occupancy_grid[:, -n:] = 1

        # Clear the starting corner (bottom-left n x n cells) so the
        # robot's own start box isn't marked occupied by the border
        # inflation - inflation exists to keep the robot's body away from
        # arena walls/obstacles it must drive around, not from the space
        # it's deliberately parked in at t=0.
        #
        # FIXED DURING PORT: the teammate's original carve-out only cleared
        # a single-cell-wide L-shaped notch (`grid[2, 2:8]=0` /
        # `grid[2:8, 2]=0`), not the corner itself - confirmed by direct
        # testing (collide_with_point(5, 5) was True, and HybridAStar found
        # zero valid moves out of a start pose placed there, since neither
        # the start cell nor most cells reachable from it in one 5cm/24-bin
        # discretised step happened to land exactly on that 1-cell notch).
        # A start pose anywhere in the arena's actual start box needs the
        # whole box open, not a razor-thin corridor out of it.
        self.occupancy_grid[:n, :n] = 0

        last = GRID_SIZE - 1
        for obstacle in obstacles:
            i_start = max(obstacle.x_g - n, 0)
            i_end = min(obstacle.x_g + n + 1, last)
            j_start = max(obstacle.y_g - n, 0)
            j_end = min(obstacle.y_g + n + 1, last)
            self.occupancy_grid[i_start:i_end + 1, j_start:j_end + 1] = 1

    def collide_with_point(self, x: float, y: float) -> bool:
        """True if the continuous-cm point (x, y) falls in an occupied cell
        or off the grid entirely."""
        x_g, y_g = coords_to_grid(x, y)
        if x_g < 0 or x_g >= GRID_SIZE or y_g < 0 or y_g >= GRID_SIZE:
            return True
        return bool(self.occupancy_grid[x_g, y_g])
