"""Smoke tests for the mdp_bringup test scaffolding.

These exist to prove the harness itself works: that `colcon test` discovers this
directory for an ament_cmake package, that pytest and hypothesis are both
importable from the pixi environment, and that the fixtures resolve real paths in
the source tree rather than the install copies. No production behaviour is
asserted here.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st


def test_pytest_discovers_this_directory():
    """Baseline: the file ran at all, so discovery and wiring work."""
    assert True


def test_hypothesis_is_importable_and_runs():
    """hypothesis is installed and its strategies are usable."""
    assert st.floats is not None


@settings(max_examples=25, deadline=None)
@given(st.floats(min_value=-4 * math.pi, max_value=4 * math.pi))
def test_hypothesis_drives_a_property(yaw):
    """A trivial property, to confirm example generation actually executes.

    Placeholder for the real yaw-normalisation property in task 4, which will
    target the pure pose helper described in README.md.
    """
    wrapped = math.atan2(math.sin(yaw), math.cos(yaw))
    assert -math.pi - 1e-9 <= wrapped <= math.pi + 1e-9


def test_source_tree_fixtures_resolve(package_root, scripts_dir):
    """Fixtures point at the source package, and the runners are where we think.

    Guards against a test run that has silently picked up `install/mdp_bringup`.
    """
    assert package_root.name == 'mdp_bringup'
    assert (package_root / 'CMakeLists.txt').is_file()
    assert 'install' not in package_root.parts
    assert (scripts_dir / 'task1_runner.py').is_file()


def test_launch_and_config_assets_exist(package_root):
    """The launch files and config the frame fix will touch are present."""
    assert (package_root / 'launch' / 'task1_sim.launch.py').is_file()
    assert (package_root / 'launch' / 'real.launch.py').is_file()
    assert (package_root / 'config' / 'ekf.yaml').is_file()
