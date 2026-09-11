#!/usr/bin/env python3
"""Pure 2D rigid pose transform math - no ROS, no numpy, standard library only.

A pose here is a plain `(x, y, yaw)` tuple: a planar rigid body transform, and
equally the pose of one frame expressed in another. Yaw is radians, wrapped to
`[-pi, pi]` on every output so the `+/-pi` seam never leaks into a comparison.

## Why this lives in `mdp_bringup` and not `mdp_algorithm`

`mdp_algorithm` is the planner, and it works in ONE coordinate system: arena
centimetres. It has no notion of a TF frame, and it must keep it that way - a
test (`test_frame_preservation.test_mdp_algorithm_has_no_frame_awareness`) walks
the AST of every module in that package and fails on a frame-name string literal
(`'map'`, `'odom'`, `'arena'`, `'base_footprint'`) or a `tf2_*` import. Frames
are a deployment concern: which frame the arena is drawn in, and where `odom` was
created, depends on how the robot was launched, not on how a route is planned.

`mdp_bringup` owns that concern. It launches the nodes, declares the start pose,
and broadcasts `map -> odom`, so the transform math that relates the arena frame
to the dead-reckoning frame belongs here, next to the task runners that use it.
Keeping the math in this module rather than inside `task1_runner` also keeps it
importable without `rclpy`, which is what lets the property tests exercise it
with no ROS graph running.

Installed to `lib/mdp_bringup` alongside `task1_runner.py`, so at runtime it is
importable as a sibling module (`import pose_transform`); the test suite gets the
same import by putting `scripts/` on `sys.path` (see `test/conftest.py`).
"""

import math

__all__ = ['normalise_yaw', 'compose', 'invert', 't_map_odom_from_start_pose']


def normalise_yaw(yaw):
    """Wrap `yaw` into `[-pi, pi]`.

    Via `atan2(sin, cos)` rather than modular arithmetic: it is exact at the
    seam and needs no special-casing of the sign of the input.
    """
    return math.atan2(math.sin(yaw), math.cos(yaw))


def compose(a, b):
    """Pose `b`, given in frame `a`, expressed in the frame `a` itself is in.

    Read as transform multiplication `a * b`. With `a = T_map_odom` and `b` a
    pose in `odom`, the result is that pose in `map`.
    """
    ax, ay, ayaw = a
    bx, by, byaw = b
    return (ax + bx * math.cos(ayaw) - by * math.sin(ayaw),
            ay + bx * math.sin(ayaw) + by * math.cos(ayaw),
            normalise_yaw(ayaw + byaw))


def invert(a):
    """The inverse transform of `a`, so `compose(invert(a), compose(a, p)) == p`.

    Rotating the negated translation by `-yaw`, written out rather than built
    from `compose` to keep it a single pass of trig.
    """
    ax, ay, ayaw = a
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (-(ax * c + ay * s), ax * s - ay * c,
            normalise_yaw(-ayaw))


def t_map_odom_from_start_pose(start_pose):
    """`T_map_odom` for a robot whose arena-frame start pose is `start_pose`.

    `odom` is created at the spawn / power-on pose with identity orientation, so
    the robot's `odom`-frame pose at that instant is `(0, 0, 0)` and

        T_map_odom = start_pose * (odom pose at creation)^-1 = start_pose

    numerically. The composition is written out anyway so the derivation is the
    code rather than a comment, and so the identity start pose falls out as the
    identity transform instead of being asserted.

    Returns a `(x, y, yaw)` tuple with yaw normalised - the translation and
    rotation a static `map -> odom` broadcaster should publish.
    """
    x, y, yaw = start_pose
    odom_pose_at_creation = (0.0, 0.0, 0.0)
    return compose((float(x), float(y), float(yaw)),
                   invert(odom_pose_at_creation))
