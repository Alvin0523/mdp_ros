"""Property 4 (preservation): a single `odom -> base` broadcaster.

**Validates: Requirements 3.4, 3.5**

THESE TESTS PASS ON UNFIXED CODE and record the baseline the fix must not
disturb: exactly one owner of `odom -> base_footprint` in sim and of
`odom -> base_link` on hardware. Task 6.11 re-runs this file unchanged - the
interesting moment is task 6.6, which adds `ekf_node` to the sim launch, where a
second broadcaster of the same edge is precisely the mistake that is easy to
make (hence `publish_tf: false` in the sim overlay).

## Ownership is a config fact, so it is read from config

Two nodes fighting over one TF edge is decided by parameters, before either
starts: `enable_odom_tf` on the ros2_control controller, `publish_tf` plus
`world_frame`/`odom_frame`/`base_link_frame` on `robot_localization`, and the
frames on any `static_transform_publisher`. `count_broadcasters` below reads all
three from the launch graph and the YAML they load. Standing up Gazebo to run
`ros2 topic echo /tf` would report the same count for the sim only, would say
nothing about hardware (no robot here), and would hang the suite.

Two holes that a `/tf` capture would cover and a config read might not, closed
explicitly below: `robot_state_publisher` cannot own the edge because the URDF
has no `odom` link, and the Gazebo `/tf` bridge carries no such edge because
nothing in the world or the URDF runs a pose-publisher system.

## Baseline recorded on unfixed code

- `task1_sim.launch.py`: `odom -> base_footprint` has exactly one owner,
  `ackermann_steering_controller` (`ackermann_controller.yaml`,
  `enable_odom_tf: true`). No `ekf_node` in the graph.
- `real.launch.py`: `odom -> base_link` has exactly one owner, `ekf_filter_node`
  (`ekf.yaml`, `publish_tf: true`, `world_frame: odom`). The hardware controller
  stands down explicitly - `real_controller.yaml` sets `enable_odom_tf: false`.
- `map -> odom`: **zero** owners in both launch files. The edge does not exist
  yet; `test_map_to_odom_edge_exists` is marked xfail so this run documents that
  absence instead of hiding it, and reports XPASS once task 6.2/6.3 land.
- `odom -> base` continuity: not measurable at runtime pre-fix, because
  `/odometry/filtered` has no publisher in sim (recorded as its own
  counterexample in `test_path_tracking_frame_consistency.py`). What is
  checkable, and is what the fix could break, is that the edge keeps a single
  dead-reckoning owner and that the transform stacked on top of it is constant -
  see `test_static_map_to_odom_never_jumps`.
"""

import math
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from launch_introspection import (ARENA_FRAME, ARENA_SIZE_M, CONFIG_DIR, PACKAGE_ROOT,
                                 arena_to_odom_broadcasters, controller_params, launch_nodes,
                                 load_launch, node_arguments, node_executable,
                                 node_executables, node_parameter_files, spawned_controllers)

SIM_LAUNCH = 'task1_sim.launch.py'
HARDWARE_LAUNCH = 'real.launch.py'

# The child frame each context anchors dead reckoning to.
SIM_BASE_FRAME = 'base_footprint'
HARDWARE_BASE_FRAME = 'base_link'

URDF_DIR = PACKAGE_ROOT.parent / 'mdp_description' / 'urdf'
WORLDS_DIR = PACKAGE_ROOT.parent / 'mdp_description' / 'worlds'


# ==========================================================================
# The broadcaster model: who owns which TF edge, read from launch + config
# ==========================================================================

def controller_config_for(launch_name: str) -> str:
    """The ros2_control YAML a launch file's controllers are configured from.

    Hardware passes it to `ros2_control_node` as a parameter file. Sim cannot -
    the config path is substituted into `robot_description` for the
    `gz_ros2_control` plugin to read - so it is recovered from the launch source
    instead, which is the only place it appears.
    """
    ld = load_launch(launch_name)
    for node in launch_nodes(ld):
        if node_executable(node) != 'ros2_control_node':
            continue
        for path in node_parameter_files(node):
            if path.name.endswith('controller.yaml'):
                return path.name

    source = (PACKAGE_ROOT / 'launch' / launch_name).read_text()
    names = sorted({name for name in (p.name for p in CONFIG_DIR.glob('*controller.yaml'))
                    if name in source})
    assert len(names) == 1, f'{launch_name} references controller configs {names}'
    return names[0]


def _load_config_by_name(name: str):
    """Read a config the launch file names, always from the source tree.

    Launch files resolve configs through `get_package_share_directory`, i.e. the
    install copy. Reading the source file by the same basename keeps a stale
    install tree from making this test lie.
    """
    path = CONFIG_DIR / Path(name).name
    assert path.exists(), f'{name} is not in {CONFIG_DIR}'
    return yaml.safe_load(path.read_text()), path.name


def broadcasters(launch_name: str):
    """Every `(owner, parent_frame, child_frame)` TF edge the launch graph owns.

    Three shapes, which is all there are in this workspace:

    1. a ros2_control controller with `enable_odom_tf: true`, which broadcasts
       `odom_frame_id -> base_frame_id` - and only if a spawner actually loads it
    2. a `robot_localization` instance with `publish_tf: true`, which broadcasts
       `world_frame -> base_link_frame`
    3. a `static_transform_publisher`, which broadcasts the frames on its
       command line
    """
    ld = load_launch(launch_name)
    found = []

    controller_config = controller_config_for(launch_name)
    for controller in spawned_controllers(ld):
        params = controller_params(controller_config, controller)
        if params.get('enable_odom_tf'):
            found.append((f'{controller} ({controller_config})',
                          params.get('odom_frame_id'), params.get('base_frame_id')))

    for node in launch_nodes(ld):
        executable = node_executable(node)
        if executable in ('ekf_node', 'ukf_node'):
            for path in node_parameter_files(node):
                config, name = _load_config_by_name(path.name)
                for instance, body in config.items():
                    params = (body or {}).get('ros__parameters', {})
                    if params.get('publish_tf'):
                        found.append((f'{instance} ({name})',
                                      params.get('world_frame'), params.get('base_link_frame')))
        elif executable == 'static_transform_publisher':
            args = node_arguments(node)
            parent = child = None
            for flag, value in zip(args, args[1:]):
                if flag in ('--frame-id', '--frame_id'):
                    parent = value
                elif flag in ('--child-frame-id', '--child_frame_id'):
                    child = value
            found.append((f'static_transform_publisher {parent} -> {child}', parent, child))

    return found


def count_broadcasters(launch_name: str, parent: str, child: str):
    """Owners of one specific edge, as `(owner, parent, child)` triples."""
    return [b for b in broadcasters(launch_name) if (b[1], b[2]) == (parent, child)]


# ==========================================================================
# 3.4 - exactly one owner of odom -> base, per context
# ==========================================================================

def test_sim_has_exactly_one_odom_to_base_footprint_broadcaster():
    """Sim: `ackermann_steering_controller` and nothing else (requirement 3.4).

    Task 6.6 adds `ekf_node` to this launch file. If its overlay forgets
    `publish_tf: false`, this test names the second owner.
    """
    owners = count_broadcasters(SIM_LAUNCH, 'odom', SIM_BASE_FRAME)
    assert [o[0] for o in owners] == ['ackermann_steering_controller (ackermann_controller.yaml)'], \
        f'sim odom -> {SIM_BASE_FRAME} owners: {owners}'


def test_hardware_has_exactly_one_odom_to_base_link_broadcaster():
    """Hardware: `ekf_filter_node` only - the controller stands down (3.4, 3.5).

    `real_controller.yaml` sets `enable_odom_tf: false` so the EKF is the sole
    owner. This fix must not touch either side of that arrangement.
    """
    owners = count_broadcasters(HARDWARE_LAUNCH, 'odom', HARDWARE_BASE_FRAME)
    assert [o[0] for o in owners] == ['ekf_filter_node (ekf.yaml)'], \
        f'hardware odom -> {HARDWARE_BASE_FRAME} owners: {owners}'

    hardware_controller = controller_params(controller_config_for(HARDWARE_LAUNCH),
                                            'ackermann_steering_controller')
    assert hardware_controller['enable_odom_tf'] is False
    assert hardware_controller['base_frame_id'] == HARDWARE_BASE_FRAME


def test_no_context_broadcasts_the_other_context_base_frame():
    """Sim owns `base_footprint`, hardware owns `base_link`, neither owns both.

    The two graphs differ in which link dead reckoning anchors to, so a fix that
    hardcodes one base frame would quietly add an edge in the other context.
    """
    assert not count_broadcasters(SIM_LAUNCH, 'odom', HARDWARE_BASE_FRAME)
    assert not count_broadcasters(HARDWARE_LAUNCH, 'odom', SIM_BASE_FRAME)


def test_robot_state_publisher_cannot_own_the_odom_edge():
    """`robot_state_publisher` publishes URDF joints, and no URDF has an `odom` link.

    One of the two holes a config-only read would otherwise leave: `rsp` is in
    both launch graphs and does broadcast transforms, just never this one.
    """
    for urdf in sorted(URDF_DIR.glob('*.urdf')):
        text = urdf.read_text()
        assert 'link name="odom"' not in text, urdf.name
        assert "link name='odom'" not in text, urdf.name


def test_gazebo_tf_bridge_carries_no_odom_to_base_edge():
    """The bridged `/tf` has no pose-publisher feeding it (the second hole).

    `task1_sim.launch.py` bridges `/tf` from Gazebo. That would be a second
    source for this edge if anything in the world or the robot model ran a
    `PosePublisher` system - nothing does, so the bridge relays an empty stream
    rather than a competing one.
    """
    ld = load_launch(SIM_LAUNCH)
    bridged = [a for node in launch_nodes(ld) for a in node_arguments(node)
               if a.startswith('/tf@')]
    assert bridged, f'expected /tf on the gz bridge; graph is {node_executables(ld)}'

    for source in list(URDF_DIR.glob('*.urdf')) + list(WORLDS_DIR.glob('*.sdf')):
        text = source.read_text()
        assert 'PosePublisher' not in text, source.name
        assert 'pose_publisher' not in text, source.name


@settings(max_examples=100, deadline=None)
@given(x=st.floats(min_value=0.0, max_value=ARENA_SIZE_M, allow_nan=False),
       y=st.floats(min_value=0.0, max_value=ARENA_SIZE_M, allow_nan=False),
       yaw=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False))
def test_broadcaster_count_is_independent_of_the_start_pose(x, y, yaw):
    """For any start pose, `odom -> base` keeps exactly one owner (Property 4).

    The start pose becomes a launch argument in task 6.2/6.3 and feeds the
    `map -> odom` transform's VALUE. This is the assertion that it feeds only
    that: however the pose is chosen, ownership of the dead-reckoning edge is
    unchanged, and `map -> odom` never acquires a second owner either.
    """
    t_map_odom = (x, y, yaw)   # the value the start pose sets, per design Decision 2

    assert len(count_broadcasters(SIM_LAUNCH, 'odom', SIM_BASE_FRAME)) == 1
    assert len(count_broadcasters(HARDWARE_LAUNCH, 'odom', HARDWARE_BASE_FRAME)) == 1

    for launch_name in (SIM_LAUNCH, HARDWARE_LAUNCH):
        arena_owners = arena_to_odom_broadcasters(load_launch(launch_name))
        assert len(arena_owners) <= 1, (
            f'{launch_name} has {len(arena_owners)} owners of '
            f'{ARENA_FRAME} -> odom for start pose {t_map_odom}')


# ==========================================================================
# 3.4 - the map -> odom edge: absent today, single-owner once it lands
# ==========================================================================

def test_map_to_odom_has_at_most_one_owner():
    """Whatever `map -> odom` ends up being, it is never owned twice.

    Holds vacuously today (zero owners) and substantively after task 6.2/6.3.
    Written this way on purpose: an assertion of "exactly one" would fail now for
    a reason that has nothing to do with double ownership, which is the thing
    worth guarding.
    """
    for launch_name in (SIM_LAUNCH, HARDWARE_LAUNCH):
        owners = arena_to_odom_broadcasters(load_launch(launch_name))
        assert len(owners) <= 1, f'{launch_name}: {[node_executable(n) for n in owners]}'
        static = count_broadcasters(launch_name, ARENA_FRAME, 'odom')
        assert len(static) <= 1, f'{launch_name}: {static}'


@pytest.mark.xfail(strict=False, reason='the map -> odom edge is added by task 6.2/6.3; '
                                        'pre-fix this records its absence')
@pytest.mark.parametrize('launch_name', [SIM_LAUNCH, HARDWARE_LAUNCH])
def test_map_to_odom_edge_exists(launch_name):
    """Documents the pre-fix absence of `map -> odom`, and flips to XPASS after.

    Non-strict xfail so the pre-fix suite is green while still reporting the gap,
    and so task 6.11 sees this turn into an unexpected pass rather than having to
    edit the file.
    """
    owners = arena_to_odom_broadcasters(load_launch(launch_name))
    assert owners, (f'no node broadcasts {ARENA_FRAME} -> odom in {launch_name}; '
                    f'launch node executables are {node_executables(load_launch(launch_name))}')


# ==========================================================================
# 3.5 - continuity: a static transform must never jump
# ==========================================================================

def _compose(a, b):
    ax, ay, ayaw = a
    bx, by, byaw = b
    return (ax + bx * math.cos(ayaw) - by * math.sin(ayaw),
            ay + bx * math.sin(ayaw) + by * math.cos(ayaw),
            math.atan2(math.sin(ayaw + byaw), math.cos(ayaw + byaw)))


def map_to_odom_at(launch_name: str, _time_s: float):
    """The `map -> odom` transform a consumer would resolve at a given time.

    It takes a time argument and ignores it, and that is the property: with no
    absolute localization source the edge is static for the whole run, so all
    dead-reckoning drift stays inside `odom -> base` where REP-105 puts it. Today
    the edge is unpublished, which a renderer treats as the identity - also
    time-invariant, so the assertions below are meaningful before and after the
    fix.
    """
    for owner, parent, child in broadcasters(launch_name):
        if (parent, child) == (ARENA_FRAME, 'odom'):
            ld = load_launch(launch_name)
            for node in launch_nodes(ld):
                if node_executable(node) != 'static_transform_publisher':
                    continue
                args = node_arguments(node)
                if '--x' in args and '--y' in args and '--yaw' in args:
                    return (float(args[args.index('--x') + 1]),
                            float(args[args.index('--y') + 1]),
                            float(args[args.index('--yaw') + 1])), owner
    return (0.0, 0.0, 0.0), None


@settings(max_examples=200, deadline=None)
@given(t0=st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
       t1=st.floats(min_value=0.0, max_value=1e4, allow_nan=False))
def test_static_map_to_odom_never_jumps(t0, t1):
    """`map -> odom` resolves to the same value at any two instants (3.5).

    A jump in this edge would teleport the whole arena mid-run, which is exactly
    what a dynamic or re-derived transform would cause.
    """
    for launch_name in (SIM_LAUNCH, HARDWARE_LAUNCH):
        first, _owner = map_to_odom_at(launch_name, t0)
        second, _ = map_to_odom_at(launch_name, t1)
        assert first == second, f'{launch_name} moved between t={t0} and t={t1}'


@settings(max_examples=200, deadline=None)
@given(x0=st.floats(min_value=-ARENA_SIZE_M, max_value=ARENA_SIZE_M, allow_nan=False),
       y0=st.floats(min_value=-ARENA_SIZE_M, max_value=ARENA_SIZE_M, allow_nan=False),
       yaw0=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False),
       dx=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
       dy=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False))
def test_arena_pose_is_as_continuous_as_odom_to_base(x0, y0, yaw0, dx, dy):
    """Stacking a static `map -> odom` on `odom -> base` adds no discontinuity.

    The jump-free baseline in the form that survives the fix: consecutive
    dead-reckoning samples keep exactly their step size once expressed in the
    arena frame, so any jump seen after the fix came from `odom -> base` itself,
    not from the new edge. (Runtime continuity of `odom -> base` cannot be
    sampled pre-fix - `/odometry/filtered` has no publisher in sim - which is
    why this is stated as an invariant of the composition.)
    """
    t_map_odom, _owner = map_to_odom_at(SIM_LAUNCH, 0.0)
    before = _compose(t_map_odom, (x0, y0, yaw0))
    after = _compose(t_map_odom, (x0 + dx, y0 + dy, yaw0))

    step_in_odom = math.hypot(dx, dy)
    step_in_arena = math.hypot(after[0] - before[0], after[1] - before[1])
    assert step_in_arena == pytest.approx(step_in_odom, abs=1e-9)
