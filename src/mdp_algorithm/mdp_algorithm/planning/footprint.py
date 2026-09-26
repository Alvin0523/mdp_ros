#!/usr/bin/env python3
"""
Exact body-footprint collision check for the planner (2026-09-25).

Replaces the old single-point test (the car's centre against 10 cm inflated
cells), which ignored the car's 23 x 18.6 cm body: a nose or tail could hit a block
while the check said clear. Here the body is a rectangle in the rear-axle frame
(planning_constants.FOOTPRINT_*), the obstacles are their true 10 x 10 cm squares,
both grown by OBSTACLE_PAD_CM, and the two are tested with the separating-axis
theorem. The table edge is checked against the body's bounding box. Units:
centimetres, radians, the same frame as the OccupancyMap.
"""

import math
from typing import Iterable

from ..common.planning_constants import (
    FOOTPRINT_FRONT_CM, FOOTPRINT_HALF_WIDTH_CM, FOOTPRINT_REAR_CM,
    OBSTACLE_PAD_CM, TABLE_EDGE_MARGIN_CM)
from .occupancy_map import ARENA_SIZE_CM, OBSTACLE_SIZE_CM

HALF_LEN = (FOOTPRINT_FRONT_CM + FOOTPRINT_REAR_CM) / 2.0
CENTRE_AHEAD = (FOOTPRINT_FRONT_CM - FOOTPRINT_REAR_CM) / 2.0   # body centre, ahead of the rear axle
HALF_W = FOOTPRINT_HALF_WIDTH_CM
_BODY_RADIUS = math.hypot(HALF_LEN, HALF_W)
_BLOCK_HALF = OBSTACLE_SIZE_CM / 2.0 + OBSTACLE_PAD_CM
_REACH_SQ = (_BODY_RADIUS + _BLOCK_HALF * math.sqrt(2.0)) ** 2


def pose_collides(obstacles: Iterable, x: float, y: float, theta: float) -> bool:
    """True if the body at rear-axle pose (x, y, theta) overlaps an obstacle block
    (plus pad) or sticks out of the table (minus edge margin)."""
    c, s = math.cos(theta), math.sin(theta)
    cx, cy = x + c * CENTRE_AHEAD, y + s * CENTRE_AHEAD

    ex = abs(c) * HALF_LEN + abs(s) * HALF_W     # half extents of the body's bounding box
    ey = abs(s) * HALF_LEN + abs(c) * HALF_W
    lo, hi = TABLE_EDGE_MARGIN_CM, ARENA_SIZE_CM - TABLE_EDGE_MARGIN_CM
    if cx - ex < lo or cx + ex > hi or cy - ey < lo or cy + ey > hi:
        return True

    ext_along = _BLOCK_HALF * (abs(c) + abs(s))  # block's half extent on the body's axes
    for o in obstacles:
        dx, dy = o.x_cm - cx, o.y_cm - cy
        if dx * dx + dy * dy > _REACH_SQ:
            continue
        if abs(dx) > ex + _BLOCK_HALF or abs(dy) > ey + _BLOCK_HALF:
            continue                              # separated on a world axis
        if abs(dx * c + dy * s) > HALF_LEN + ext_along or abs(-dx * s + dy * c) > HALF_W + ext_along:
            continue                              # separated on a body axis
        return True
    return False
