"""Property 3 (bug condition): path tracking is frame-consistent.

**Validates: Requirements 1.5, 2.5**

THIS TEST IS EXPECTED TO FAIL ON UNFIXED CODE. It is the control-side instance of
the frame bug: `task1_runner` plans in arena coordinates and then steers off an
`odom`-frame pose, so the tracking error at `t=0` contains a constant rotation and
translation rather than being zero. After task 6.5/6.6 it must pass unchanged.

## Two separate defects, two separate assertions

1. **`/odometry/filtered` has no publisher in the Task 1 sim** (root cause 4).
   `real.launch.py` runs `ekf_node`; `task1_sim.launch.py` runs no EKF and no
   relay, so `odom_callback` never fires and `current_pose` sits at its
   constructor default forever. Until that is fixed, tracking error is not merely
   rotated, it is meaningless. Asserted structurally against both launch files -
   who publishes a topic is a launch-graph fact, and standing up Gazebo to run
   `ros2 topic info /odometry/filtered` would confirm the same zero-publisher
   count at 100x the cost.
2. **The pose the follower consumes is in the wrong frame** (root cause 3).
   The planned first waypoint of `leg_paths[0]` in arena metres versus the pose
   the runner actually feeds `PurePursuitController.update_pose()`, obtained by
   driving the live `odom_callback` with one `odom`-frame odometry message (see
   "Oracle corrected" below). `odom` is created at the spawn pose with identity
   orientation, so the robot's `odom`-frame pose at `t=0` is `(0, 0, 0)`, and on
   unfixed code that is passed straight through as if it were an arena pose.
   Design.md Decision 5's relay stopgap
   (`/ackermann_steering_controller/odometry` -> `/odometry/filtered`) is what
   would deliver that message on unfixed code; the message is injected directly
   here rather than the relay being launched, so defect 1 does not mask defect 2.

## Counterexamples recorded on unfixed code

Run: `pixi run pytest src/mdp_bringup/test/test_path_tracking_frame_consistency.py`
-> 4 failed, 2 passed. The two that pass are the hardware contrast case
(`real.launch.py` does run `ekf_node`) and the relay-stopgap scope guard.

1. `test_sim_launch_publishes_odometry_filtered`:
   "/odometry/filtered has no publisher in task1_sim.launch.py; launch node
   executables are ['create', 'parameter_bridge', 'publish_test_obstacles.py',
   'robot_state_publisher', 'spawner', 'task1_runner.py', 'yolo_detector.py']" -
   no `ekf_node`, no relay. This is the structural form of
   `ros2 topic info /odometry/filtered` showing zero publishers.
2. `test_runner_pose_is_not_pinned_to_its_constructor_default`:
   "task1_runner.current_pose stays at its constructor default
   (0.0, 0.0, 1.570796326795) for the whole sim run: /odometry/filtered has no
   publisher, so odom_callback never fires". The pose the control loop reports over
   Bluetooth and feeds the follower is a constant, not a measurement.
3. `test_tracking_error_at_start_is_zero_for_the_reported_pose`, on real planner
   output for `config/test_obstacles.yaml` (visiting order `[0, 1, 4, 2, 3]`, first
   checkpoint `(0.500, 0.800, -3.142)`):
   "heading error 1.5708 rad (90.0 deg): follower pose (0.0, 0.0, 0.0) (odom frame)
   vs leg 0 first waypoint (0.150, 0.200, 1.571) (arena frame)". Position error
   `hypot(0.150, 0.200) = 0.250 m`. Design.md anticipated about 0.21 m, which is
   `hypot(0.15, 0.15)` against the raw `(0.15, 0.15)` planning start pose; the
   observed 0.250 m is larger because Hybrid A*'s first node snaps `y` to the
   planner's 10 cm grid. Same defect, slightly larger number.
4. `test_tracking_error_at_start_is_zero_for_any_start_pose`, explicit example
   `(x=0.15, y=0.15, yaw=1.5707963267948966)`: same 1.5708 rad error, and the
   widened generator fails for every start pose other than exactly `(0, 0, 0)` -
   tracking is frame-consistent only at the arena origin with zero yaw.

## Oracle corrected during task 6 (test-design bug, not a weakened assertion)

`pose_the_follower_consumes` was a test-design bug of the same kind as the two
corrected in `test_arena_frame_agreement.py`: it asserted a property of the
pre-fix fixture instead of a property of the code. It hardcoded the pre-fix
behaviour -

    odom_frame_pose_at_t0 = (0.0, 0.0, 0.0)
    follower = PurePursuitController()
    follower.update_pose(*odom_frame_pose_at_t0)

- and never called `task1_runner` at all, so it returned `(0, 0, 0)` no matter
what the implementation did. `heading_err < 1e-6` was therefore unsatisfiable by
construction for any start pose but the identity, and no correct fix could have
turned it green: it was measuring a model of the bug, not the behaviour.

It now exercises the real path. A `Task1Runner` is constructed in-process
(`rclpy.init()`, no spin, no Gazebo - the `TappedRunner` pattern from
`test_frame_preservation.py`), its `tf_buffer` is seeded with the static
`map -> odom` transform the launch files broadcast (derived from the start pose
through the production helper, `pose_transform.t_map_odom_from_start_pose`), and
one `/odometry/filtered` message stamped in `odom` placing the robot at the `odom`
origin at `t=0` is fed to the real `odom_callback`. What comes back is
`follower.current_pose` - the pose the steering law actually reads - so the
`PoseStamped` wrap, the TF lookup, `do_transform_pose` and the yaw extraction are
all under test. Sanity check of that path: with `T_map_odom = (0.15, 0.15, pi/2)`
and an `odom` pose of `(1, 0, 0)`, `current_pose` comes back `(0.15, 1.15, pi/2)`.

What each test is FOR is unchanged: at `t=0` the follower's pose must coincide
with the first waypoint of the arena-frame planned path - for the reported start
pose (against real leg-0 planner output, the same fixture as before) and for any
start pose. `test_pose_is_not_updated_when_the_arena_transform_is_missing` covers
the branch a passing tracking assertion cannot distinguish: on a failed lookup
the pose must be kept, never replaced with the raw `odom` pose.

One tolerance moved with the oracle, for the same reason. In the concrete case the
waypoint is real Hybrid A* output, and its first expanded node snaps `y` to the
planner's 10 cm grid: leg 0 from `(0.15, 0.15, pi/2)` starts at `(0.15, 0.2, pi/2)`
(the same value `test_frame_preservation.BASELINE_LEG0` records). A frame-correct
pose therefore sits 0.05 m from that waypoint, so `position_err < 1e-6` was a
second thing no correct implementation could satisfy - it was asserting the
planner has infinite resolution. The position tolerance is now one grid cell
(`occupancy_map.CELL_SIZE_CM`), and the exact claim is made where it actually
holds: the pose the follower reads must equal the arena start pose to `1e-9`.
Pre-fix that pose was `(0, 0, 0)`, 0.25 m and `pi/2` away, so both assertions
still fail on unfixed code. The widened property test needs no such allowance -
leg 0's first waypoint IS the leg's start pose there by construction, so it keeps
`1e-6` on both.
"""

import ast
import math
from pathlib import Path

import pytest
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import Quaternion, TransformStamped
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from nav_msgs.msg import Odometry

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node as LaunchNode

from mdp_algorithm.planning.collision_aware_planner import plan_leg, plan_visiting_order
from mdp_algorithm.planning.occupancy_map import CELL_SIZE_CM

from pose_transform import t_map_odom_from_start_pose

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# task1_runner.TASK1_CAMERA_THETA_OFFSET_RAD - the camera faces the car's left.
CAMERA_THETA_OFFSET_RAD = -math.pi / 2.0

REPORTED_START_POSE = (0.15, 0.15, math.pi / 2)
ARENA_SIZE_M = 2.0

# The frame `/odometry/filtered` reports in - its contract, unchanged by the fix.
ODOM_FRAME = 'odom'

# `occupancy_map.CELL_SIZE_CM` in metres. Hybrid A*'s first expanded node lands on
# this grid, so leg 0 starts within one cell of the start pose rather than exactly
# on it - see the position tolerance in
# `test_tracking_error_at_start_is_zero_for_the_reported_pose`.
PLANNER_CELL_SIZE_M = CELL_SIZE_CM / 100.0

# Anything that would give `/odometry/filtered` a publisher.
ODOM_FILTERED_PUBLISHERS = ('ekf_node', 'relay', 'topic_tools')


# --------------------------------------------------------------------------
# Launch-description introspection (no Gazebo, no ROS graph)
# --------------------------------------------------------------------------

def load_launch(name: str) -> LaunchDescription:
    """Load a launch file from the SOURCE tree, without visiting it."""
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    return get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / 'launch' / name))


def launch_nodes(ld: LaunchDescription):
    return [a for a in ld.entities if isinstance(a, LaunchNode)]


def node_executable(node: LaunchNode) -> str:
    raw = getattr(node, '_Node__node_executable', None) or getattr(node, 'node_executable', '')
    try:
        return ''.join(part.text for part in raw)
    except (TypeError, AttributeError):
        return str(raw)


def odometry_filtered_publishers(ld: LaunchDescription):
    return [node_executable(n) for n in launch_nodes(ld)
            if node_executable(n) in ODOM_FILTERED_PUBLISHERS]


def runner_constructor_default_pose():
    """`self.current_pose = (...)` as assigned in `Task1Runner.__init__`.

    Read off the AST rather than instantiated, so no ROS context is needed.
    """
    tree = ast.parse((PACKAGE_ROOT / 'scripts' / 'task1_runner.py').read_text())
    for func in ast.walk(tree):
        if not (isinstance(func, ast.FunctionDef) and func.name == '__init__'):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == 'current_pose':
                    return tuple(round(v, 12) for v in
                                 eval(compile(ast.Expression(node.value), '<pose>', 'eval'),
                                      {'math': math}))
    return None


# --------------------------------------------------------------------------
# Real planner output, and the pose the follower actually consumes
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def first_leg_waypoint():
    """The first waypoint of `leg_paths[0]`, from the real planner on the real
    test layout. Hybrid A* is the expensive call (a few seconds), so this runs
    once per module.
    """
    cfg = yaml.safe_load((PACKAGE_ROOT / 'config' / 'test_obstacles.yaml').read_text())
    obstacles_grid = [(o['x'] * 100.0, o['y'] * 100.0, o['facing']) for o in cfg['obstacles']]

    _order, checkpoints, _unreachable, occ_map = plan_visiting_order(
        obstacles_grid, REPORTED_START_POSE, theta_offset=CAMERA_THETA_OFFSET_RAD)
    leg0 = plan_leg(occ_map, REPORTED_START_POSE, checkpoints[0],
                    theta_offset=CAMERA_THETA_OFFSET_RAD)
    assert leg0, 'planner found no path for leg 0 - cannot check tracking error'
    return tuple(float(v) for v in leg0[0])


@pytest.fixture(scope='module')
def runner():
    """A real `Task1Runner`, in-process: `rclpy.init()`, no spin, no Gazebo.

    Same pattern as `test_frame_preservation.TappedRunner`. Nothing is spun, so
    the 20 Hz control-loop timer never fires and the node is driven purely by
    calling its callbacks directly. Constructed once per module - the node's TF
    buffer is replaced per case (see `pose_the_follower_consumes`), which is the
    only per-case state that matters here.
    """
    rclpy.init()
    import task1_runner
    node = task1_runner.Task1Runner()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def static_map_odom_transform(arena_frame: str, start_pose) -> TransformStamped:
    """The `map -> odom` edge the launch files broadcast for `start_pose`.

    Derived through the production helper, so this fixture cannot disagree with
    what the running system publishes.
    """
    x, y, yaw = t_map_odom_from_start_pose(start_pose)
    tf = TransformStamped()
    tf.header.frame_id = arena_frame
    tf.child_frame_id = ODOM_FRAME
    tf.transform.translation.x = float(x)
    tf.transform.translation.y = float(y)
    tf.transform.rotation = quaternion_from_yaw(yaw)
    return tf


def odometry_at_odom_origin() -> Odometry:
    """`/odometry/filtered` at `t=0`: the robot at the `odom` origin, stamped `odom`.

    `odom` is created at the spawn/power-on pose with identity orientation, so
    this - not the arena pose - is genuinely what the odometry source reports at
    `t=0`. Position and orientation are the message defaults (zero translation,
    identity quaternion); the stamp is left at zero, which a static transform
    resolves regardless.
    """
    msg = Odometry()
    msg.header.frame_id = ODOM_FRAME
    return msg


def pose_the_follower_consumes(node, start_pose, seed_tf=True):
    """The pose the REAL `odom_callback` hands the follower at `t=0`.

    Exercises the production path end to end, with only the two things a running
    system would supply substituted in: the `map -> odom` edge the launch file
    broadcasts (seeded as a static transform, derived from `start_pose` via
    `pose_transform.t_map_odom_from_start_pose`) and one `/odometry/filtered`
    message. Everything between - the `PoseStamped` wrap, the TF lookup, the
    `do_transform_pose` call, the yaw extraction, `follower.update_pose` - is
    `task1_runner`'s own code.

    `seed_tf=False` leaves the buffer empty to exercise the lookup-failure path.
    Returns `follower.current_pose`, i.e. what the steering law actually reads.
    """
    node.tf_buffer = tf2_ros.Buffer()
    if seed_tf:
        node.tf_buffer.set_transform_static(
            static_map_odom_transform(node.arena_frame, start_pose), 'test_fixture')
    node.odom_callback(odometry_at_odom_origin())
    return node.follower.current_pose


def yaw_error(actual: float, expected: float) -> float:
    return abs(math.atan2(math.sin(actual - expected), math.cos(actual - expected)))


# --------------------------------------------------------------------------
# Defect 1 - the blocker: /odometry/filtered has no publisher in sim
# --------------------------------------------------------------------------

def test_sim_launch_publishes_odometry_filtered():
    """The sim must give `/odometry/filtered` a publisher (requirement 2.5).

    EXPECTED FAILURE: zero publishers, so `odom_callback` never fires.
    """
    ld = load_launch('task1_sim.launch.py')
    publishers = odometry_filtered_publishers(ld)
    executables = sorted(node_executable(n) for n in launch_nodes(ld))
    assert publishers, (
        '/odometry/filtered has no publisher in task1_sim.launch.py; '
        f'launch node executables are {executables}'
    )


def test_hardware_launch_publishes_odometry_filtered():
    """Contrast case, passes on unfixed code: hardware does run the EKF.

    Pins that the sim/hardware graphs disagree, which is the actual defect - not
    that `/odometry/filtered` is unused everywhere.
    """
    ld = load_launch('real.launch.py')
    assert 'ekf_node' in odometry_filtered_publishers(ld)


def test_runner_pose_is_not_pinned_to_its_constructor_default():
    """With no odometry publisher, `current_pose` never leaves its default.

    EXPECTED FAILURE: the constructor default is `(0.0, 0.0, pi/2)` and, in sim,
    nothing ever overwrites it - a constant pose, not a measurement.
    """
    default_pose = runner_constructor_default_pose()
    assert default_pose is not None, 'could not find self.current_pose in Task1Runner.__init__'

    sim_has_odom = bool(odometry_filtered_publishers(load_launch('task1_sim.launch.py')))
    assert sim_has_odom, (
        f'task1_runner.current_pose stays at its constructor default {default_pose} '
        'for the whole sim run: /odometry/filtered has no publisher, so odom_callback '
        'never fires'
    )


# --------------------------------------------------------------------------
# Defect 2 - tracking error at t=0 - EXPECTED TO FAIL
# --------------------------------------------------------------------------

def test_tracking_error_at_start_is_zero_for_the_reported_pose(runner, first_leg_waypoint):
    """The follower's pose and the planned path must share a frame (requirement 2.5).

    Scoped to the reported start pose and the real `test_obstacles.yaml` layout:
    the waypoint is real Hybrid A* output for `config/test_obstacles.yaml`, and
    the pose is what the live `odom_callback` produces.

    ORACLE CORRECTED in task 6 - see the module docstring, "Oracle corrected"
    (including why the position tolerance here is a grid cell and not `1e-6`).

    EXPECTED FAILURE on unfixed code: heading error about `pi/2`, position error
    about 0.25 m (see counterexample 3 in the module docstring).
    """
    consumed = pose_the_follower_consumes(runner, REPORTED_START_POSE)
    wx, wy, wyaw = first_leg_waypoint

    # The exact claim: the pose the follower reads IS the arena start pose. No
    # tolerance is needed for this half - it is the frame relation itself, and it
    # is what carries the strength of this test.
    assert yaw_error(consumed[2], REPORTED_START_POSE[2]) < 1e-9, (
        f'follower pose {consumed} is not the arena start pose '
        f'{REPORTED_START_POSE}')
    assert math.hypot(consumed[0] - REPORTED_START_POSE[0],
                      consumed[1] - REPORTED_START_POSE[1]) < 1e-9, (
        f'follower pose {consumed} is not the arena start pose '
        f'{REPORTED_START_POSE}')

    heading_err = yaw_error(consumed[2], wyaw)
    position_err = math.hypot(consumed[0] - wx, consumed[1] - wy)

    assert heading_err < 1e-6, (
        f'heading error {heading_err:.4f} rad ({math.degrees(heading_err):.1f} deg): '
        f'follower pose {consumed} (as odom_callback produced it) vs leg 0 first '
        f'waypoint ({wx:.3f}, {wy:.3f}, {wyaw:.3f}) (arena frame)'
    )
    assert position_err < PLANNER_CELL_SIZE_M, (
        f'position error {position_err:.3f} m between follower pose {consumed} and '
        f'leg 0 first waypoint ({wx:.3f}, {wy:.3f}); more than the planner grid '
        f'cell ({PLANNER_CELL_SIZE_M} m), so this is a frame offset and not the '
        f'grid snap'
    )


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(x=REPORTED_START_POSE[0], y=REPORTED_START_POSE[1], yaw=REPORTED_START_POSE[2])
@given(x=st.floats(min_value=0.0, max_value=ARENA_SIZE_M, allow_nan=False),
       y=st.floats(min_value=0.0, max_value=ARENA_SIZE_M, allow_nan=False),
       yaw=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False))
def test_tracking_error_at_start_is_zero_for_any_start_pose(runner, x, y, yaw):
    """Widened form: no start pose may introduce a constant tracking offset.

    A leg planned from arena pose `(x, y, yaw)` begins at that pose, so at `t=0`
    the follower should already be on its first waypoint. Hybrid A* is not re-run
    per example - the first waypoint of leg 0 IS the leg's start pose by
    construction (confirmed against the real planner in
    `test_tracking_error_at_start_is_zero_for_the_reported_pose`), so this is the
    pure frame relation, without a multi-second search per example. The pose side
    is still the live `odom_callback`, per start pose.

    ORACLE CORRECTED in task 6 - see the module docstring, "Oracle corrected".

    EXPECTED FAILURE on unfixed code for every start pose other than exactly
    `(0, 0, 0)`.
    """
    planned_first_waypoint = (x, y, yaw)
    consumed = pose_the_follower_consumes(runner, planned_first_waypoint)

    heading_err = yaw_error(consumed[2], planned_first_waypoint[2])
    position_err = math.hypot(consumed[0] - planned_first_waypoint[0],
                              consumed[1] - planned_first_waypoint[1])

    assert heading_err < 1e-6, (
        f'heading error {heading_err:.4f} rad for arena start pose '
        f'({x:.3f}, {y:.3f}, {yaw:.3f}); follower consumed {consumed}'
    )
    assert position_err < 1e-6, (
        f'position error {position_err:.3f} m for arena start pose '
        f'({x:.3f}, {y:.3f}, {yaw:.3f})'
    )


def test_pose_is_not_updated_when_the_arena_transform_is_missing(runner):
    """With no `map -> odom` in TF, the update is DROPPED, not applied raw.

    The other half of requirement 2.5, and the one a passing tracking test cannot
    see: a raw fallback would look correct whenever TF happened to be up and
    silently reintroduce the 90 degree error whenever it was not - intermittent,
    and invisible in the tracking numbers. So the contract is stale-over-wrong,
    and the pose the follower holds must be untouched by an unresolvable message.
    """
    sentinel = (7.0, -3.0, 0.25)
    runner.follower.update_pose(*sentinel)
    runner.current_pose = sentinel

    consumed = pose_the_follower_consumes(runner, REPORTED_START_POSE, seed_tf=False)

    assert consumed == sentinel, (
        f'follower pose moved to {consumed} with no {runner.arena_frame} -> '
        f'{ODOM_FRAME} transform available; the raw odom pose was used as an '
        f'arena pose'
    )
    assert runner.current_pose == sentinel, (
        f'node.current_pose moved to {runner.current_pose} on a failed lookup')


def test_relay_stopgap_is_not_in_the_launch_graph():
    """Scope guard, passes on unfixed code and after the fix.

    Design.md Decision 5 allows the relay only as a test fixture. The odometry
    message it would have carried is injected directly in
    `pose_the_follower_consumes`, the relay itself is never launched, and task 6.6
    replaces it with `ekf_node` + `ekf_sim.yaml`. This fails if someone ships the
    stopgap.
    """
    ld = load_launch('task1_sim.launch.py')
    relays = [node_executable(n) for n in launch_nodes(ld)
              if node_executable(n) in ('relay', 'topic_tools')]
    assert not relays, f'relay stopgap shipped in task1_sim.launch.py: {relays}'
    assert not [a for a in ld.entities
                if isinstance(a, DeclareLaunchArgument) and a.name == 'relay_odometry']
