#!/usr/bin/env python3
"""
Geometry helpers shared by hybrid_astar.py, reeds_shepp_curves.py and
hamiltonian.py. Ported from the teammate's mdp_algo package (utils.py) -
distance/angle-normalisation helpers only, pygame/matplotlib-dependent
pieces dropped.

BUG FIXED DURING PORT: the original change_of_basis() ran
`theta1 = deg_to_rad(p1[2])` on p1[2], and every reeds_shepp path1..path12
function then did the same `deg_to_rad(phi)` on their own phi argument -
but every caller in this codebase (Hamiltonian, HybridAStar, Node.theta)
passes theta already in RADIANS (e.g. np.pi/2), never degrees. Converting
an already-radian value through deg_to_rad() a second time silently
shrinks it by a factor of ~57 (multiplying by pi/180 again), corrupting
both the coordinate rotation in change_of_basis() and the phi angle in
every Reeds-Shepp path formula - which would have made the Reeds-Shepp
heuristic (and Hamiltonian's optional 'reeds-shepp' distance metric)
silently wrong for any non-trivial heading. Confirmed by tracing the exact
call chain (get_all_paths -> change_of_basis -> path1..path12) rather than
assumed - fixed by removing the spurious deg_to_rad() calls; angles now
stay in radians throughout, matching every other module in this package.
"""

import math
from typing import Tuple


def facing_to_rad(facing: str) -> float:
    """{'N','S','E','W'} -> radians, 0=East/CCW+ (REP-103 convention)."""
    assert facing in ('N', 'S', 'E', 'W')
    return {'E': 0.0, 'N': math.pi / 2, 'W': math.pi, 'S': -math.pi / 2}[facing]


def l1(x1: float, y1: float, x2: float, y2: float) -> float:
    """L1 (Manhattan) distance between 2 points in R2."""
    return abs(x1 - x2) + abs(y1 - y2)


def l2(x1: float, y1: float, x2: float, y2: float) -> float:
    """L2 (Euclidean) distance between 2 points in R2."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def diag_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Octile/diagonal distance - admissible heuristic for 8-connected grids."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return math.sqrt(2 * min(dx, dy) ** 2) + abs(dx - dy)


def M(theta: float) -> float:
    """Wrap angle (radians) to [-pi, pi)."""
    theta = theta % (2 * math.pi)
    if theta < -math.pi:
        return theta + 2 * math.pi
    if theta >= math.pi:
        return theta - 2 * math.pi
    return theta


def R(x: float, y: float) -> Tuple[float, float]:
    """Polar coordinates (r, theta) of the point (x, y)."""
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta


def change_of_basis(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Position/heading of p2=(x2,y2,theta2) expressed in the frame whose
    origin is p1=(x1,y1) and whose x-axis points along theta1. All thetas in
    RADIANS (see module docstring for the deg/rad bug this fixes)."""
    theta1 = p1[2]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    new_x = dx * math.cos(theta1) + dy * math.sin(theta1)
    new_y = -dx * math.sin(theta1) + dy * math.cos(theta1)
    new_theta = p2[2] - p1[2]
    return new_x, new_y, new_theta
