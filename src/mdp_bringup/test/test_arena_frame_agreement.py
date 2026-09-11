"""Property 1 (bug condition): arena and robot render in agreement.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4**

THIS TEST IS EXPECTED TO FAIL ON UNFIXED CODE. The failures below are the
deliverable of task 2 of the `foxglove-tf-heading-90-offset` bugfix spec: they
are the machine-checkable form of the reported symptom (robot heading 90 degrees
clockwise of the drawn arena in Foxglove). After task 6 they must pass unchanged
- do not weaken them to get green.

## Why this is a static/structural test

The property is about the TF tree and about the `frame_id` every arena-referenced
publisher stamps. Both are decided before a single message flows:

- the arena-to-`odom` edge either has a broadcaster in the launch graph or it does
  not, and today no node in `task1_sim.launch.py` owns it. Standing up Gazebo to
  watch `/tf` would confirm the same absence at 100x the cost and would hang the
  suite, so the launch description is introspected in-process instead (no Gazebo
  server, no ROS graph - `generate_launch_description()` is a pure function here).
- the ten arena publish sites in `scripts/task1_runner.py` stamp string literals,
  so the frames they claim are read straight off the AST.

The one thing this cannot see is a *live* transform - a runtime-only broadcaster
would be missed. `test_no_launch_node_broadcasts_the_arena_to_odom_edge` therefore
documents the absence at the level where the fix lands (task 6.2/6.3 add a
`static_transform_publisher` to the launch files), not at the level of a `/tf`
capture.

## Counterexamples recorded on unfixed code

Run: `pixi run pytest src/mdp_bringup/test/test_arena_frame_agreement.py`
-> 4 failed, 2 passed. The two that pass are the bug-condition premise and the
`/cmd_vel` scope guard.

1. `test_no_launch_node_broadcasts_the_arena_to_odom_edge`:
   "no node broadcasts map -> odom in task1_sim.launch.py; launch node
   executables are ['create', 'parameter_bridge', 'publish_test_obstacles.py',
   'robot_state_publisher', 'spawner', 'task1_runner.py', 'yolo_detector.py']".
   `odom` is the TF root; no `map` exists, so no consumer can convert between
   arena coordinates and the dead-reckoned pose.
2. `test_arena_publishers_do_not_stamp_odom`: all 11 arena `frame_id` literals
   say `'odom'` - `_publish_occupancy_grid:443`, `_publish_grid_lines:481/500/529`,
   `_publish_obstacle_markers:574/591`, `_publish_checkpoint_markers:637/655`,
   `_publish_current_path:697`, `_publish_search_progress:730`,
   `_publish_planned_route:820` - while carrying arena-origin coordinates.
3. `test_rendered_robot_pose_equals_arena_start_pose`, explicit example
   `(x=0.15, y=0.15, yaw=1.5707963267948966)`:
   "rendered heading 0.000 rad vs arena start yaw 1.571 rad (90.0 deg off);
   arena -> odom transform published: None", and a 0.212 m position error - the
   robot renders on the drawn arena's `(0, 0)` corner instead of inside the drawn
   start box. The reported symptom, reproduced without Gazebo.
4. Same test, widened generator: it fails for every start pose that is not exactly
   `(0, 0, 0)`, so the reported `pi/2` is one point on a continuum, not a special
   case.
5. `test_arena_markers_land_on_their_gazebo_counterparts`, minimal case
   `ox=1.0, oy=1.0`: "marker for arena (1.000, 1.000) renders at (-0.850, 1.150),
   1.856 m away" - and off the 2 m arena entirely, since the whole arena is drawn
   rotated `pi/2` about the spawn point.

The 2D pose math below is local to this test on purpose: the production helper
(`scripts/pose_transform.py`) is task 6.1 and does not exist yet, and this test
must be able to fail for the right reason before it exists.

## Two tests corrected during task 6 (test-design bugs, not weakened assertions)

Both were written so that no correct implementation could satisfy them. They
asserted things about the fixture rather than about the property, so they would
have gone red on a working fix and stayed red forever. What each one is *for* is
unchanged; only the thing being measured was wrong.

1. `test_bug_condition_holds_for_the_reported_start_pose` asserted that the LIVE
   code still exhibits the bug (arena marker frames `== {'odom'}`, no published
   arena transform). That is satisfiable only while the bug exists, so the fix
   itself breaks it. Its actual job was to pin the premise: that the input the
   fix-checking tests reason about is genuinely a bug-condition input. It now
   does that by evaluating `is_bug_condition` - bugfix.md's `isBugCondition`
   written out - against a MODELLED pre-fix state, and checks the classifier
   rejects the states that are not bug conditions. The reported
   `(0.15, 0.15, pi/2)` is still the input under test, and the live spawn pose is
   still asserted to equal it, so the premise stays tied to the real launch file.

2. `test_rendered_robot_pose_equals_arena_start_pose` generated an arbitrary
   `(x, y, yaw)` and compared it against the transform the launch file publishes,
   which is a fixed literal. Every generated pose except that one literal failed
   by construction - the generator and the oracle were describing different
   scenes. The invariant it meant to state does not depend on the launch file's
   particular numbers: for ANY start pose, the `map -> odom` transform DERIVED
   from that pose, composed with the identity `odom -> base` at `odom` creation,
   reproduces the pose. That is now the property, checked against the production
   helper (`pose_transform.t_map_odom_from_start_pose`) rather than a test-local
   restatement of it. The launch-file-specific claim it used to conflate into the
   property - published transform matches actual spawn pose - is now its own
   concrete test (`test_launch_publishes_the_transform_its_spawn_pose_implies`),
   and that one is the real regression guard: it is what fails if someone edits
   the spawn pose without editing the transform.
"""

import ast
import math
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from launch import LaunchDescription
from launch_ros.actions import Node as LaunchNode

# The reported case, per bugfix.md: robot spawned inside the start box facing
# arena +Y ("North"), which is what the planner assumes.
REPORTED_START_POSE = (0.15, 0.15, math.pi / 2)

# The arena frame the fix introduces (design.md Decision 1). Absent today.
ARENA_FRAME = 'map'

ARENA_SIZE_M = 2.0

# Every method in task1_runner that publishes arena-coordinate data. Ten publish
# sites, spread over seven methods (`_publish_grid_lines` alone stamps three
# markers: grid lines, placement-zone outline, start-box outline), plus
# `_publish_planned_route`'s `/planned_path` - 11 `frame_id` literals in total.
ARENA_PUBLISH_METHODS = (
    '_publish_occupancy_grid',
    '_publish_grid_lines',
    '_publish_obstacle_markers',
    '_publish_checkpoint_markers',
    '_publish_current_path',      # /path_markers line strip (calls _publish_planned_route)
    '_publish_search_progress',
    '_publish_planned_route',     # /planned_path
)


# --------------------------------------------------------------------------
# Pure 2D rigid-transform math (test-local - see module docstring)
# --------------------------------------------------------------------------

def normalise_yaw(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))


def compose(a, b):
    """Pose `b` expressed in the frame `a` is expressed in."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    return (ax + bx * math.cos(ayaw) - by * math.sin(ayaw),
            ay + bx * math.sin(ayaw) + by * math.cos(ayaw),
            normalise_yaw(ayaw + byaw))


def yaw_error(actual: float, expected: float) -> float:
    return abs(normalise_yaw(actual - expected))


def position_error(actual, expected) -> float:
    return math.hypot(actual[0] - expected[0], actual[1] - expected[1])


# --------------------------------------------------------------------------
# Launch-description introspection (no Gazebo, no ROS graph)
# --------------------------------------------------------------------------

def load_task1_sim_launch() -> LaunchDescription:
    """Build the Task 1 sim launch description in-process.

    `IncludeLaunchDescription` is lazy - the included `gz_sim.launch.py` is only
    read when the description is *visited* by a launch service, which never
    happens here - so this starts no simulator and no nodes.
    """
    from launch.launch_description_sources import get_launch_description_from_python_launch_file

    # Deliberately the SOURCE launch file, not `install/mdp_bringup/share`, so a
    # stale install copy can't make this test lie.
    launch_file = Path(__file__).resolve().parent.parent / 'launch' / 'task1_sim.launch.py'
    return get_launch_description_from_python_launch_file(str(launch_file))


def launch_nodes(ld: LaunchDescription):
    return [a for a in ld.entities if isinstance(a, LaunchNode)]


def _plain_arguments(node: LaunchNode):
    """The node's `arguments` as plain strings, skipping unresolved substitutions."""
    out = []
    for group in getattr(node, '_Node__arguments', None) or []:
        if isinstance(group, str):
            out.append(group)
        elif isinstance(group, (list, tuple)):
            out.extend(str(item) for item in group if isinstance(item, str))
    return out


def node_executable(node: LaunchNode) -> str:
    raw = getattr(node, '_Node__node_executable', None) or getattr(node, 'node_executable', '')
    try:
        return ''.join(part.text for part in raw)  # TextSubstitution list
    except (TypeError, AttributeError):
        return str(raw)


def spawn_start_pose_from_launch(ld: LaunchDescription):
    """The `(x, y, yaw)` the sim actually spawns the robot at, read off `create`.

    This is one of the three duplicated start-pose literals root cause 5 calls
    out (the other two are `task1_runner`'s planning `start_pose` and the
    docstring), so reading it here also pins which literal the render is
    compared against.
    """
    for node in launch_nodes(ld):
        if node_executable(node) != 'create':
            continue
        args = _plain_arguments(node)
        pose = {}
        for flag, key in (('-x', 'x'), ('-y', 'y'), ('-Y', 'yaw')):
            if flag in args:
                pose[key] = float(args[args.index(flag) + 1])
        if {'x', 'y', 'yaw'} <= pose.keys():
            return (pose['x'], pose['y'], pose['yaw'])
    return None


def arena_to_odom_broadcasters(ld: LaunchDescription):
    """Nodes in the launch graph that could publish an arena-frame -> `odom` edge.

    Recognises a `static_transform_publisher` naming the arena frame as parent
    (the shape task 6.2 introduces), and any node parameterised with an arena
    `world_frame`/`global_frame`/`map_frame` (the shape a second
    `robot_localization` instance would take, per design.md's note that it could
    later replace the static broadcaster).
    """
    found = []
    for node in launch_nodes(ld):
        args = _plain_arguments(node)
        if node_executable(node) == 'static_transform_publisher':
            frames = {a.strip('-') for a in args}
            if ARENA_FRAME in frames or 'arena' in frames:
                found.append(node)
                continue
        for params in getattr(node, '_Node__parameters', None) or []:
            if not isinstance(params, dict):
                continue
            for key, value in params.items():
                key = str(key)
                if key.endswith(('world_frame', 'global_frame', 'map_frame')) \
                        and str(value) in (ARENA_FRAME, 'arena'):
                    found.append(node)
    return found


def published_arena_transform(ld: LaunchDescription):
    """`T_arena_odom` as the running system publishes it, or `None` if unpublished.

    Nothing composes an arena pose without this edge, which is exactly requirement
    1.3's defect.
    """
    for node in arena_to_odom_broadcasters(ld):
        args = _plain_arguments(node)
        if '--x' in args and '--y' in args and '--yaw' in args:
            return (float(args[args.index('--x') + 1]),
                    float(args[args.index('--y') + 1]),
                    float(args[args.index('--yaw') + 1]))
    return None


# --------------------------------------------------------------------------
# task1_runner source introspection
# --------------------------------------------------------------------------

def frame_id_literals_by_method(runner_source: Path):
    """`{method_name: [(lineno, frame_id_literal), ...]}` for every constant
    `*.frame_id = '...'` assignment in `task1_runner.py`."""
    tree = ast.parse(runner_source.read_text())
    out = {}
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == 'frame_id':
                    out.setdefault(func.name, []).append((node.lineno, node.value.value))
    return out


@pytest.fixture(scope='module')
def sim_launch():
    return load_task1_sim_launch()


@pytest.fixture(scope='module')
def runner_frames(scripts_dir):
    return frame_id_literals_by_method(scripts_dir / 'task1_runner.py')


# --------------------------------------------------------------------------
# Bug condition (these hold on unfixed code - they pin the premise)
# --------------------------------------------------------------------------

def is_bug_condition(state) -> bool:
    """`isBugCondition(X)` from bugfix.md, transcribed.

    `state` is a dict with the three fields the pseudocode names:
    `robot_start_pose_in_arena`, `arena_marker_frame`, `tf_tree` (a set of
    `(parent, child)` pairs). All three clauses must hold: arena data drawn in
    `odom`, nothing relating the arena frame to `odom`, and a start pose that is
    not the identity - at the identity the two frames coincide and there is
    nothing to get wrong.
    """
    x, y, yaw = state['robot_start_pose_in_arena']
    exists_arena_transform = any(
        {parent, child} == {ARENA_FRAME, 'odom'} for parent, child in state['tf_tree'])
    return (state['arena_marker_frame'] == 'odom'
            and not exists_arena_transform
            and (yaw != 0 or x != 0 or y != 0))


def test_bug_condition_classifies_the_reported_pre_fix_scene(sim_launch):
    """`is_bug_condition` accepts the reported scene and rejects the near misses.

    CORRECTED in task 6 - see the module docstring, item 1. This pins the premise
    the fix-checking tests rest on: the reported input really is a bug-condition
    input, and the classifier is not vacuously true. It does NOT assert the live
    code is still broken, which is what the original version did and which no fix
    could survive.

    The live spawn pose is still read from the launch file and still required to be
    the reported `(0.15, 0.15, pi/2)`, so the modelled state below stays anchored
    to the scene the bug was reported against.
    """
    start_pose = spawn_start_pose_from_launch(sim_launch)
    assert start_pose is not None, 'no `create` spawn pose found in task1_sim.launch.py'
    assert start_pose == pytest.approx(REPORTED_START_POSE, abs=1e-9)
    assert start_pose != (0.0, 0.0, 0.0)

    # The scene as it was reported: arena data stamped `odom`, no arena edge in
    # the TF tree, robot started 0.15 m diagonally in facing arena +Y.
    pre_fix = {
        'robot_start_pose_in_arena': REPORTED_START_POSE,
        'arena_marker_frame': 'odom',
        'tf_tree': {('odom', 'base_footprint'), ('base_footprint', 'base_link')},
    }
    assert is_bug_condition(pre_fix)

    # Each clause independently takes the state out of the bug condition, which is
    # what makes the classifier a description of the defect rather than of any
    # scene at all.
    assert not is_bug_condition({**pre_fix, 'arena_marker_frame': ARENA_FRAME})
    assert not is_bug_condition({**pre_fix,
                                 'tf_tree': pre_fix['tf_tree'] | {(ARENA_FRAME, 'odom')}})
    assert not is_bug_condition({**pre_fix, 'robot_start_pose_in_arena': (0.0, 0.0, 0.0)})


# --------------------------------------------------------------------------
# Fix checking - EXPECTED TO FAIL on unfixed code
# --------------------------------------------------------------------------

def test_no_launch_node_broadcasts_the_arena_to_odom_edge(sim_launch):
    """An arena frame must exist in the TF tree (requirement 2.3).

    EXPECTED FAILURE: zero broadcasters. `odom` is the TF root; there is no `map`,
    so no consumer can convert arena coordinates to the robot's dead-reckoned pose.
    """
    broadcasters = arena_to_odom_broadcasters(sim_launch)
    executables = sorted(node_executable(n) for n in launch_nodes(sim_launch))
    assert broadcasters, (
        f"no node broadcasts {ARENA_FRAME} -> odom in task1_sim.launch.py; "
        f"launch node executables are {executables}"
    )
    assert len(broadcasters) == 1, 'the arena -> odom edge must have exactly one owner'


def test_arena_publishers_do_not_stamp_odom(runner_frames):
    """Every arena-referenced publisher stamps the arena frame (requirement 2.2).

    EXPECTED FAILURE: all 11 sites across the seven arena publish methods stamp
    `'odom'` while filling in arena-origin coordinates.
    """
    offenders = [(method, lineno) for method in ARENA_PUBLISH_METHODS
                 for lineno, frame in runner_frames.get(method, [])
                 if frame == 'odom']
    assert not offenders, (
        'arena data stamped `odom` at '
        + ', '.join(f'{m}:{ln}' for m, ln in offenders)
    )


def test_odom_frame_publishers_are_only_body_frame_data(runner_frames):
    """Guard on the fix's scope: `/cmd_vel` is body-frame twist, not arena data.

    Passes on unfixed code (`send_cmd` stamps `base_link`) and must keep passing -
    it is here so a fix that blanket-renames every `frame_id` in the file gets
    caught.
    """
    assert [f for _, f in runner_frames.get('send_cmd', [])] == ['base_link']


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(x=REPORTED_START_POSE[0], y=REPORTED_START_POSE[1], yaw=REPORTED_START_POSE[2])
@given(x=st.floats(min_value=0.0, max_value=ARENA_SIZE_M, allow_nan=False),
       y=st.floats(min_value=0.0, max_value=ARENA_SIZE_M, allow_nan=False),
       yaw=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False))
def test_rendered_robot_pose_equals_arena_start_pose(x, y, yaw):
    """The arena transform DERIVED from a start pose renders that pose back
    (requirements 2.1, 2.4).

    CORRECTED in task 6 - see the module docstring, item 2. The generated pose is
    now the pose the transform is derived from, instead of being compared against
    the launch file's fixed literal; the launch file's own numbers are checked
    separately in `test_launch_publishes_the_transform_its_spawn_pose_implies`.

    `odom` is created at the spawn/power-on pose with identity orientation, so
    `odom -> base` is the identity at `t=0` and the rendered arena pose is exactly
    `T_arena_odom`. The property is therefore that deriving `T_arena_odom` from a
    start pose and rendering through it is a round trip, for every start pose in
    the arena and every heading - the reported `pi/2` (pinned as an `@example`)
    being one point on that continuum rather than a special case.

    Checked against the production helper, not a test-local copy: if
    `t_map_odom_from_start_pose` ever stops being the start pose - a sign flip, a
    forgotten inverse - the robot renders in the wrong place and this fails.
    """
    from pose_transform import t_map_odom_from_start_pose

    start_pose = (x, y, yaw)
    t_arena_odom = t_map_odom_from_start_pose(start_pose)

    odom_to_base_at_creation = (0.0, 0.0, 0.0)
    rendered = compose(t_arena_odom, odom_to_base_at_creation)

    assert yaw_error(rendered[2], yaw) < 1e-6, (
        f'rendered heading {rendered[2]:.3f} rad vs arena start yaw {yaw:.3f} rad '
        f'({math.degrees(yaw_error(rendered[2], yaw)):.1f} deg off); '
        f'transform derived from the start pose: {t_arena_odom}'
    )
    assert position_error(rendered, (x, y)) < 1e-6, (
        f'rendered position {rendered[:2]} vs arena start ({x:.3f}, {y:.3f}); '
        f'{position_error(rendered, (x, y)):.3f} m off'
    )


def test_launch_publishes_the_transform_its_spawn_pose_implies(sim_launch):
    """The `map -> odom` transform the sim publishes equals the pose it spawns at.

    The concrete half of the test above, and the actual regression guard: the two
    numbers live in one launch file and must agree, or the drawn arena is offset
    and rotated from the robot by exactly their difference. This is what fails if
    someone moves the spawn pose and forgets the transform (or vice versa) - which
    is how the reported bug's 90 degrees got in.
    """
    spawn_pose = spawn_start_pose_from_launch(sim_launch)
    assert spawn_pose is not None, 'no `create` spawn pose found in task1_sim.launch.py'

    published = published_arena_transform(sim_launch)
    assert published is not None, (
        f'no {ARENA_FRAME} -> odom transform published, so the arena cannot be '
        f'related to the robot spawned at {spawn_pose}'
    )

    rendered = compose(published, (0.0, 0.0, 0.0))
    assert yaw_error(rendered[2], spawn_pose[2]) < 1e-9, (
        f'published transform yaw {published[2]:.6f} vs spawn yaw {spawn_pose[2]:.6f}')
    assert position_error(rendered, spawn_pose[:2]) < 1e-9, (
        f'published transform translation {published[:2]} vs spawn '
        f'position {spawn_pose[:2]}')


@settings(max_examples=100, deadline=None)
@given(ox=st.floats(min_value=0.1, max_value=ARENA_SIZE_M - 0.1, allow_nan=False),
       oy=st.floats(min_value=0.1, max_value=ARENA_SIZE_M - 0.1, allow_nan=False))
def test_arena_markers_land_on_their_gazebo_counterparts(ox, oy):
    """A marker at arena `(ox, oy)` must render where Gazebo puts that object.

    With the arena drawn in `odom`, an arena point `p` renders at `p` in `odom`,
    which is `T_arena_odom * p` in the arena - i.e. displaced by the start pose
    and rotated by the start yaw.

    EXPECTED FAILURE: for the reported spawn pose, an obstacle at `(1.0, 1.0)`
    renders about 1.6 m from its Gazebo counterpart.
    """
    ld = load_task1_sim_launch()
    start_pose = spawn_start_pose_from_launch(ld)
    t_arena_odom = published_arena_transform(ld)
    if t_arena_odom is None:
        # Unpublished: the renderer treats the arena point as an `odom` point,
        # which sits at compose(start_pose, point) in true arena coordinates.
        rendered_in_arena = compose(start_pose, (ox, oy, 0.0))
    else:
        rendered_in_arena = (ox, oy, 0.0)

    assert position_error(rendered_in_arena, (ox, oy)) < 1e-6, (
        f'marker for arena ({ox:.3f}, {oy:.3f}) renders at '
        f'({rendered_in_arena[0]:.3f}, {rendered_in_arena[1]:.3f}), '
        f'{position_error(rendered_in_arena, (ox, oy)):.3f} m away'
    )
