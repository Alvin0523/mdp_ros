# mdp_bringup tests

Run with `pixi run test` from the workspace root (`colcon test`), or directly
with `pixi run pytest src/mdp_bringup/test` while iterating.

`mdp_bringup` is an `ament_cmake` package, so this directory is registered
explicitly in `CMakeLists.txt` via `ament_add_pytest_test`. Adding a new file
here needs no CMake change - the whole directory is one test target.

## The pose-transform seam these tests target

The frame bug (`foxglove-tf-heading-90-offset`) is fundamentally 2D rigid-body
math: the arena frame and `odom` are related by the robot's start pose, and
nothing in the current code expresses that relation. The property tests for it
must not need a live ROS graph, a Gazebo instance or a TF listener, so the math
gets extracted into a pure helper:

    src/mdp_bringup/scripts/pose_transform.py   (NOT YET IMPLEMENTED - task 6.1)

Planned surface, all pure functions over `(x, y, yaw)` tuples in radians:

- derive `T_map_odom` from an arena start pose (`odom` is created at the
  spawn/power-on pose with identity orientation, so `T_map_odom` is numerically
  that start pose)
- compose two 2D pose transforms
- invert a 2D pose transform
- normalise yaw across the `+/-pi` wrap

`scripts/` is installed to `lib/mdp_bringup`, so `task1_runner.py` imports the
helper as a sibling module and the tests import it by path. Deliberately kept
out of `mdp_algorithm`, which must stay free of any frame concept.

Nothing in this directory implements the helper. Task 1 is scaffolding only.

## Exploration tests that are SUPPOSED to fail right now

`test_arena_frame_agreement.py` (Property 1) and
`test_path_tracking_frame_consistency.py` (Property 3) are bug-condition
exploration tests for the `foxglove-tf-heading-90-offset` spec. On unfixed code
each contributes 4 failures and 2 passes: the failures encode the expected
behaviour and are the deliverable, the passes pin the bug-condition premise and
guard the fix's scope. Do not weaken them to get a green run - they turn green
in tasks 6.8 and 6.9.

Both work at the static/structural level (launch-description introspection plus
the runner's AST) so no Gazebo instance is ever started. Each file's module
docstring records the counterexamples observed on unfixed code.
