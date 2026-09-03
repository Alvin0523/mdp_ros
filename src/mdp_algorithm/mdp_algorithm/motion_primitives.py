#!/usr/bin/env python3
"""
Gear/Steering primitives shared by hybrid_astar.py, reeds_shepp_curves.py and
hamiltonian.py. Split out to avoid import cycles between those modules.
"""

from enum import IntEnum


class Gear(IntEnum):
    FORWARD = 1
    PARK = 0
    REVERSE = -1


class Steering(IntEnum):
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1
