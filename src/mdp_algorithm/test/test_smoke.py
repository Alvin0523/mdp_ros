"""Smoke tests for the mdp_algorithm test scaffolding.

`mdp_algorithm` is an ament_python package, so `colcon test` runs pytest over
this directory automatically - no build-file registration needed beyond the
`test` extra already declared in setup.py.

Scope note: mdp_algorithm is deliberately frame-agnostic. It works in arena
coordinates and knows nothing about `map`, `odom` or TF, and the frame fix must
keep it that way. The pure pose-transform helper the frame property tests target
lives in mdp_bringup instead (see src/mdp_bringup/test/README.md). Tests here
cover planner/geometry numerics only.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from mdp_algorithm.common.geometry_utils import M, l2


def test_package_is_importable():
    """The package resolves from the test run, so later tests can exercise it."""
    import mdp_algorithm  # noqa: F401


def test_yaw_wrap_examples():
    """M() wraps to [-pi, pi) on the boundary cases that matter."""
    assert M(0.0) == 0.0
    assert math.isclose(M(math.pi), -math.pi, abs_tol=1e-9)
    assert math.isclose(M(-math.pi), -math.pi, abs_tol=1e-9)
    assert math.isclose(M(3 * math.pi / 2), -math.pi / 2, abs_tol=1e-9)


@settings(max_examples=50, deadline=None)
@given(st.floats(min_value=-8 * math.pi, max_value=8 * math.pi))
def test_yaw_wrap_lands_in_range(theta):
    """Placeholder property: M() always lands in [-pi, pi)."""
    wrapped = M(theta)
    assert -math.pi - 1e-9 <= wrapped < math.pi + 1e-9


def test_euclidean_distance_is_symmetric():
    """Sanity check that the shared geometry helper behaves."""
    assert math.isclose(l2(0.0, 0.0, 3.0, 4.0), 5.0)
    assert math.isclose(l2(3.0, 4.0, 0.0, 0.0), 5.0)
