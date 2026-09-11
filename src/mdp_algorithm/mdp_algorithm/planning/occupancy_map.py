#!/usr/bin/env python3
"""
Collision-aware occupancy grid for the 200x200cm / 20x20-cell MDP arena.

Ported from the teammate's mdp_algo package (objects/OccupancyMap.py,
objects/Obstacle.py) - pygame (sprite rendering) and matplotlib
(interactive plotting) dependencies stripped, since this only needs to run
headless on the Pi as a planning input, never rendered. All internal units
are CENTIMETRES (matching the teammate's original grid math exactly, to
avoid introducing conversion bugs into ported search code) - callers at the
mdp_algorithm package boundary (e.g. collision_aware_planner.py) are
responsible for converting to/from metres before handing paths to
pure_pursuit_follower.

REVISED 2026-09-04: the 200x200cm zone is where obstacles get placed, not a
physical wall - confirmed the robot is allowed to plan/drive outside it.
Removed the border-blocking + start-corner-carve-out that used to treat the
outer 15cm as a hard wall on all four sides (a real assumption, not a
placeholder - there IS no wall there). The occupancy array is padded by
GRID_MARGIN_CM on every side purely to give the search room to route outside
the nominal placement zone when useful; nothing in that margin is ever
marked occupied except real obstacle inflation, so it acts as open space,
not a wall moved further out. GRID_SIZE now refers to the PADDED array (see
below) - PLACEMENT_ZONE_CELLS is the original 20x20 for anything that needs
to draw/reason about just the nominal obstacle-placement area specifically
(e.g. visualization).

REVISED 2026-09-04 (again): arena grid is actually 10cm/cell, not 5cm -
confirmed by direct user correction, not derived. Also switched obstacle
placement from "snap the given coordinate down into whichever grid cell
contains it" (x_g/y_g, integer grid indices - meant the obstacle's actual
10x10cm footprint could sit up to a whole cell width away from the
coordinate it was supposedly placed at) to storing the obstacle's true
CONTINUOUS centre position (x_cm/y_cm) directly - the physical convention is
that a given placement coordinate (e.g. "1m, 1m") IS the centre of the
10x10cm block, not a grid-cell corner it gets rounded into. Inflation
(add_obstacles_to_grid, below) now checks true Euclidean distance from that
continuous centre, not a fixed square of grid cells - see INFLATION_RADIUS_CM.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

ARENA_SIZE_CM = 200.0   # 200x200cm nominal obstacle-placement zone (not a wall)
CELL_SIZE_CM = 10.0     # fixed independently of grid size - do not derive from ARENA_SIZE_CM/GRID_SIZE
PLACEMENT_ZONE_CELLS = int(ARENA_SIZE_CM / CELL_SIZE_CM)   # 20 - the nominal zone, cell units

# Obstacles are placed as a 10x10cm block centred exactly on the given
# coordinate - independent of CELL_SIZE_CM (the grid's own resolution),
# even though they're numerically equal right now. Used by hamiltonian.py's
# obstacle_to_checkpoint() to find the centre of whichever edge faces the
# scan side, and by add_obstacles_to_grid()'s inflation radius below.
OBSTACLE_SIZE_CM = 10.0

# How far past the nominal placement zone the robot may be planned, on every
# side - not a wall moved outward, just search headroom. Adjustable; 100cm
# was picked as a reasonable maneuvering margin, not a spec value.
GRID_MARGIN_CM = 100.0
GRID_MARGIN_CELLS = int(GRID_MARGIN_CM / CELL_SIZE_CM)   # 10

GRID_SIZE = PLACEMENT_ZONE_CELLS + 2 * GRID_MARGIN_CELLS   # 40 - the actual (padded) array size

# Keep-out radius, in cm, from an obstacle's CENTRE - a true circle (Euclidean
# distance, see add_obstacles_to_grid), not a square of grid cells like the
# old INFLATION_CELLS scheme. 20cm is a STARTING VALUE, same caveat as every
# other un-bench-verified constant in this codebase (see
# planning_constants.py): roughly the obstacle's own half-diagonal
# (OBSTACLE_SIZE_CM/2 * sqrt(2) ~= 7.07cm) plus this robot's own half-width
# clearance (~8cm, traction_track_width/2, per docs/stm32/architecture.md)
# plus a small safety pad - not independently measured/tuned yet. Tune this
# ONE constant to change the whole keep-out zone's size; checkpoint standoff
# (hamiltonian.py's obstacle_to_checkpoint) derives its preferred distance
# from it rather than hardcoding a separate number.
INFLATION_RADIUS_CM = 20.0


@dataclass
class Obstacle:
    """Continuous-centre obstacle position + the direction its scannable
    image faces.

    x_cm/y_cm: the CENTRE of the obstacle's 10x10cm (OBSTACLE_SIZE_CM)
    footprint, in continuous cm - in the SAME frame as everything else in
    this package (nominal placement zone's own (0,0), not the padded grid's
    corner - see grid_to_coords/coords_to_grid for that offset). NOT a grid
    index - a real point, so "obstacle placed at (100, 100)" means its block
    is centred exactly there, not snapped into whichever cell contains it.
    facing: {'N', 'S', 'E', 'W'} - direction the agent must face to see the
    obstacle's image, matching the teammate's convention.
    """
    x_cm: float
    y_cm: float
    facing: str
    id: int = -1


def grid_to_coords(x_g: float, y_g: float) -> Tuple[float, float]:
    """Padded-grid cell coordinates -> continuous cm coordinates (cell
    origin), in the SAME cm frame obstacles/start poses are given in (i.e.
    (0,0) here is the nominal placement zone's own corner, not the padded
    array's corner - the GRID_MARGIN_CELLS offset is added/removed here so
    every other module never has to think about the padding)."""
    return (x_g - GRID_MARGIN_CELLS) * CELL_SIZE_CM, (y_g - GRID_MARGIN_CELLS) * CELL_SIZE_CM


def coords_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Continuous cm coordinates (nominal placement-zone frame) -> padded-grid
    cell coordinates. Single source of truth for this conversion - hybrid_astar.py's
    Node._discretize() calls this too rather than re-deriving it, after a
    duplicated inline copy there (hardcoded 200/GRID_SIZE, no margin offset)
    was found during the 2026-09-04 padding change - would have silently
    produced negative/wrapped array indices for any pose in the new margin."""
    return (int(x // CELL_SIZE_CM) + GRID_MARGIN_CELLS,
            int(y // CELL_SIZE_CM) + GRID_MARGIN_CELLS)


class OccupancyMap:
    def __init__(self, obstacles: List[Obstacle] = None) -> None:
        """
        Parameters:
            occupancy_grid (np.array): GRID_SIZE x GRID_SIZE (padded) binary
                grid, 1 = occupied (inflated obstacle footprint only - no
                border/wall, see module docstring), 0 = free.
        """
        obstacles = obstacles or []
        assert len(obstacles) <= 8   # arena has at most 8 obstacles

        self.xmin, self.xmax, self.ymin, self.ymax = grid_to_coords(0, 0) + grid_to_coords(GRID_SIZE, GRID_SIZE)
        self.obstacles: List[Obstacle] = []
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE))

        self.add_obstacles_to_grid(obstacles)

    def add_obstacles_to_grid(self, obstacles: List[Obstacle]) -> None:
        assert len(self.obstacles) + len(obstacles) <= 8
        self.obstacles += obstacles

        # Circular (true Euclidean-distance) inflation from the obstacle's
        # continuous centre - replaces the old square-of-grid-cells scheme
        # (i_start=x_g-n..x_g+n), which wasn't actually a "radius" despite
        # being called INFLATION_CELLS: it blocked a square, not a circle,
        # so corner cells at (n,n) cells away were blocked even though
        # they're sqrt(2)*n*CELL_SIZE_CM from the obstacle - further than
        # cells at (n,0) that were also blocked. A cell is occupied here iff
        # the distance from ITS OWN CENTRE to the obstacle's centre is
        # within INFLATION_RADIUS_CM - a real circle.
        #
        # Only scans a bounding box around each obstacle (not the whole
        # padded grid) for efficiency - INFLATION_RADIUS_CM plus half a cell
        # of slack so no partially-covered edge cell is missed.
        last = GRID_SIZE - 1
        r = INFLATION_RADIUS_CM
        r_cells = int(np.ceil(r / CELL_SIZE_CM)) + 1
        for obstacle in obstacles:
            ox_g, oy_g = coords_to_grid(obstacle.x_cm, obstacle.y_cm)
            i_start = max(ox_g - r_cells, 0)
            i_end = min(ox_g + r_cells, last)
            j_start = max(oy_g - r_cells, 0)
            j_end = min(oy_g + r_cells, last)
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    cell_x, cell_y = grid_to_coords(i, j)
                    cell_cx = cell_x + CELL_SIZE_CM / 2.0
                    cell_cy = cell_y + CELL_SIZE_CM / 2.0
                    if (cell_cx - obstacle.x_cm) ** 2 + (cell_cy - obstacle.y_cm) ** 2 <= r ** 2:
                        self.occupancy_grid[i, j] = 1

    def collide_with_point(self, x: float, y: float) -> bool:
        """True if the continuous-cm point (x, y) falls in an occupied cell
        or off the padded array entirely (only ever hit ~100cm past the
        nominal placement zone - not a wall at the placement zone's own
        edge, see module docstring)."""
        x_g, y_g = coords_to_grid(x, y)
        h, w = self.occupancy_grid.shape
        if x_g < 0 or x_g >= h or y_g < 0 or y_g >= w:
            return True
        return bool(self.occupancy_grid[x_g, y_g])
