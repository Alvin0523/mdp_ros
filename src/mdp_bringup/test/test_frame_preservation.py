"""Property 2 (preservation): non-buggy inputs render and behave identically.

**Validates: Requirements 3.1, 3.2, 3.3, 3.6, 3.7**

THESE TESTS PASS ON UNFIXED CODE, and that is the point: every number in here was
read off the unfixed system first and then written down, so after task 6 they
either still pass (the fix touched only frames) or they name exactly which value
the fix moved. Task 6.10 re-runs this file unchanged.

## Observation-first, and what "observed" means here

The arena publishers are methods on a live `Task1Runner`, so the geometry is
observed by constructing that node in-process (`rclpy.init()`, no Gazebo, no
spin, no external process), replacing each publisher's `publish` with a capture
hook, and calling the publish methods directly. What lands in the hook IS the
message the renderer would receive - not a re-derivation of it. The recorded
digests below come from exactly the functions this file uses, run against the
unfixed tree.

The `NOT C` branch of `isBugCondition` is the identity start pose `(0, 0, 0)`:
`T_map_odom` is then the identity, arena coordinates and `odom` coordinates
coincide, and every published value must survive the fix untouched. `frame_id`
is the one field allowed to change - it starts as `'odom'` and becomes a frame
that, for this start pose, is numerically the same frame - so the assertions
below check that the arena publishers agree on ONE frame rather than that they
say any particular one.

## Baseline recorded on unfixed code

`config/test_obstacles.yaml`, camera offset `-pi/2`:

- identity start pose `(0, 0, 0)`: visiting order `[0, 1, 4, 2, 3]`, no
  unreachable obstacles, checkpoints `(0.5, 0.8, -pi)`, `(1.0, 0.4, pi/2)`,
  `(1.5, 0.7, -pi)`, `(1.7, 1.7, 0.0)`, `(1.1, 1.8, -pi/2)`
- reported start pose `(0.15, 0.15, pi/2)`: identical order, checkpoints and
  unreachable list - the start pose moves the legs, not the route
- occupancy grid: 40x40 cells at 0.1 m, origin `(-1.0, -1.0)`, 60 occupied cells
- `/obstacle_setup` in metres lands obstacle cube centres on the given
  coordinates exactly, with `#N [facing] (x, y)` labels at one decimal
- `/start_run` accepts only in `WAITING_FOR_GO`; `/yolo_result` is read only in
  `PAUSE_FOR_SCAN` and only for the first detection; Bluetooth strings are
  `TARGET,<1-based obstacle>,<id|UNKNOWN>` and `ROBOT,<x.2f>,<y.2f>,<deg.0f>`
- `config/ekf.yaml`: 9563 bytes, sha256 `1b1a73c2...c128845`

## The pose-transform helper

`scripts/pose_transform.py` is task 6.1 and does not exist yet, so the rigid-
transform properties below run against a local reference implementation and
switch to the production helper automatically once it appears (see
`_load_pose_transform`). That way this file is meaningful before the helper
exists and tests the real thing afterwards.
"""

import ast
import hashlib
import math
import struct

import pytest
import rclpy
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from std_msgs.msg import String
from std_srvs.srv import Trigger

from mdp_algorithm.planning.collision_aware_planner import plan_leg, plan_visiting_order
from mdp_algorithm.planning.occupancy_map import Obstacle, OccupancyMap

from launch_introspection import (ARENA_SIZE_M, CONFIG_DIR, PACKAGE_ROOT, REPORTED_START_POSE,
                                 load_config)

# `odom` is created at the spawn pose with identity orientation, so this start
# pose makes `T_map_odom` the identity: the `NOT C` branch of isBugCondition.
IDENTITY_START_POSE = (0.0, 0.0, 0.0)


def f32(value):
    """Snap a bound to the nearest exactly-representable 32-bit float.

    The generators below draw with `width=32` on purpose: the published values
    are compared for EXACT equality against the generated input, so the drawn
    numbers have to survive every narrowing on the way through. Hypothesis
    refuses a `width=32` strategy whose bounds are not themselves float32 values
    (`0.05`, `1.95` and `math.pi` are all float64-only), so the bounds go through
    here. Do NOT "simplify" this back to raw `math.pi` - it raises
    `InvalidArgument` at collection time.
    """
    return struct.unpack('f', struct.pack('f', value))[0]

# task1_runner.TASK1_CAMERA_THETA_OFFSET_RAD - the camera faces the car's left.
CAMERA_THETA_OFFSET_RAD = -math.pi / 2.0

MDP_ALGORITHM_ROOT = PACKAGE_ROOT.parent / 'mdp_algorithm' / 'mdp_algorithm'

# Frame names that would mean `mdp_algorithm` had grown a frame concept. Not
# `base_link`: `pure_pursuit_follower`'s standalone node stamps its own
# body-frame twist with it, which is a message-header fact, not arena awareness.
TF_FRAME_NAMES = {'map', 'odom', 'arena', 'base_footprint'}

# --- Recorded baseline: config/ekf.yaml, byte for byte (requirement 3.5) -----
EKF_YAML_SHA256 = '1b1a73c25c18dc34b4c5ab371786d341211b5846396be48bb5cf799a8c128845'
EKF_YAML_BYTES = 9563

# --- Recorded baseline: planner numerics for config/test_obstacles.yaml ------
BASELINE_VISITING_ORDER = [0, 1, 4, 2, 3]
BASELINE_UNREACHABLE = []
BASELINE_CHECKPOINTS = [
    (0.5, 0.8, -3.14159265359),
    (1.0, 0.4, 1.570796326795),
    (1.5, 0.7, -3.14159265359),
    (1.7, 1.7, 0.0),
    (1.1, 1.8, -1.570796326795),
]
BASELINE_GRID_SHAPE = (40, 40)
BASELINE_GRID_OCCUPIED_CELLS = 60
BASELINE_GRID_SHA256 = '75d9c0346dd090c98add88b97e205a062f17879b62a6bc53908f00bae20e12e2'

# --- Recorded baseline: published geometry, identity start pose --------------
# Digests of every geometry field of every message the arena publishers emit for
# the identity start pose, `frame_id` deliberately excluded (the one field the
# fix is allowed to change). Produced by `scene_geometry_digests()` below,
# against the unfixed tree.
BASELINE_SCENE_DIGESTS = {
    'occupancy_grid': 'b59193b79ac3c7b308632ba5d542590a2c52177b84ee63d13bcd872f94aefc50',
    'grid_markers': '076d3130b66c83694d4f028c2410a785659430f50e94d2a44039425ba61e88e2',
    'obstacle_markers': 'cd20d7abe3d903649fd8826b9cd26f66b30ac9073e841db9b0298f01a4142e6e',
    'checkpoint_markers': 'd73b37f992b33058a26135ae5182feef789292c7bc2f8a4177f460fedb2d342c',
    'planned_path': '3768d9296e486db73b575bd0e80355c88ba729a5586fcb7eed6b944b008a9d32',
    'path_markers': '380a0d4e8ed017a5391bc65ba87a33dde058d429312876760d29cb2edf5d52e2',
    'search_progress': '7314727eac8bf62057ee0e423fd370f8c888c8baf515af28de492750ecf9b145',
}

# Leg 0 as Hybrid A* plans it today, per start pose: (length, first waypoint).
# The reported pose's first waypoint has y snapped to the planner's 10 cm grid,
# which is why it reads 0.2 rather than 0.15.
BASELINE_LEG0 = {
    IDENTITY_START_POSE: (30, (0.05, 0.0, 0.0)),
    REPORTED_START_POSE: (24, (0.15, 0.2, 1.570796326795)),
}


# ==========================================================================
# Pure 2D rigid-transform math
# ==========================================================================

def _load_pose_transform():
    """The production helper if task 6.1 has landed, else a reference impl.

    The reference implementation is the definition of a 2D rigid transform, so
    once `scripts/pose_transform.py` exists these properties compare the real
    helper against that definition instead of against itself.
    """
    try:
        import pose_transform  # noqa: F401  (scripts/ is on sys.path via conftest)
    except ImportError:
        return _ReferencePoseTransform, False
    return pose_transform, True


class _ReferencePoseTransform:
    @staticmethod
    def normalise_yaw(yaw):
        return math.atan2(math.sin(yaw), math.cos(yaw))

    @staticmethod
    def compose(a, b):
        ax, ay, ayaw = a
        bx, by, byaw = b
        return (ax + bx * math.cos(ayaw) - by * math.sin(ayaw),
                ay + bx * math.sin(ayaw) + by * math.cos(ayaw),
                _ReferencePoseTransform.normalise_yaw(ayaw + byaw))

    @staticmethod
    def invert(a):
        ax, ay, ayaw = a
        c, s = math.cos(ayaw), math.sin(ayaw)
        return (-(ax * c + ay * s), ax * s - ay * c,
                _ReferencePoseTransform.normalise_yaw(-ayaw))


POSE_TRANSFORM, POSE_TRANSFORM_IS_PRODUCTION = _load_pose_transform()


def yaw_error(actual, expected):
    return abs(math.atan2(math.sin(actual - expected), math.cos(actual - expected)))


# ==========================================================================
# Observation: real planner output (cached - Hybrid A* costs seconds)
# ==========================================================================

def _obstacles_from_config():
    cfg = load_config('test_obstacles.yaml')
    return [(o['x'] * 100.0, o['y'] * 100.0, o['facing']) for o in cfg['obstacles']]


def _plan_route(start_pose):
    return plan_visiting_order(_obstacles_from_config(), start_pose,
                               theta_offset=CAMERA_THETA_OFFSET_RAD)


_ROUTE_CACHE = {}
_LEG_CACHE = {}


def planned_route(start_pose):
    """`plan_visiting_order` output for the real test layout, cached per start pose."""
    if start_pose not in _ROUTE_CACHE:
        _ROUTE_CACHE[start_pose] = _plan_route(start_pose)
    return _ROUTE_CACHE[start_pose]


def planned_leg0(start_pose):
    """`plan_leg` output for leg 0, cached - this is the multi-second call."""
    if start_pose not in _LEG_CACHE:
        _order, checkpoints, _unreachable, occ_map = planned_route(start_pose)
        _LEG_CACHE[start_pose] = plan_leg(occ_map, start_pose, checkpoints[0],
                                          theta_offset=CAMERA_THETA_OFFSET_RAD)
    return _LEG_CACHE[start_pose]


def grid_digest(occupancy_grid):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(occupancy_grid).tobytes()).hexdigest()


# ==========================================================================
# Observation: the live node, with its publishers tapped
# ==========================================================================

CAPTURE_TOPICS = {
    'grid_pub': 'occupancy_grid',
    'grid_marker_pub': 'grid_markers',
    'obstacle_marker_pub': 'obstacle_markers',
    'checkpoint_marker_pub': 'checkpoint_markers',
    'path_pub': 'planned_path',
    'path_marker_pub': 'path_markers',
    'search_progress_pub': 'search_progress',
    'cmd_pub': 'cmd_vel',
    'bt_pub': 'bluetooth_tx',
}


class TappedRunner:
    """A real `Task1Runner` whose publishers record instead of transmitting.

    Nothing is spun and no external process is involved: the publish methods are
    called directly and the messages they hand to `publish()` are kept. That is
    the same object a subscriber would deserialise, so "published geometry" means
    literally the published geometry.
    """

    def __init__(self):
        import task1_runner
        self.module = task1_runner
        self.node = task1_runner.Task1Runner()
        self.captured = {}
        for attr, topic in CAPTURE_TOPICS.items():
            self._tap(attr, topic)

    def _tap(self, attr, topic):
        def capture(msg, _topic=topic):
            self.captured.setdefault(_topic, []).append(msg)
        getattr(self.node, attr).publish = capture

    def reset(self):
        self.captured.clear()
        node = self.node
        node.state = self.module.State.WAITING_FOR_SETUP
        node.obstacles = []
        node.visiting_order = []
        node.checkpoints = []
        node.leg_paths = []
        node.unreachable = []
        node.current_target_idx = 0
        node.occ_map = None
        node.detected_target_id = None
        node.tablet_ids = []
        node.plan_state = 'WAITING'
        node.reset_done = False
        node.stopped = False
        node._plan_gen += 1
        node._last_sent = {}
        node._last_heartbeat = node.get_now_sec()
        node._manual_until = 0.0
        node._manual_active = False
        node.current_pose = (0.0, 0.0, math.pi / 2)
        node.follower.active = False
        node.state_start_time = node.get_now_sec()

    def one(self, topic):
        msgs = self.captured.get(topic, [])
        assert msgs, f'nothing published on {topic}'
        return msgs[-1]

    def destroy(self):
        self.node.destroy_node()


@pytest.fixture(scope='module')
def runner():
    rclpy.init()
    tapped = TappedRunner()
    try:
        yield tapped
    finally:
        tapped.destroy()
        rclpy.shutdown()


def setup_string(obstacles):
    """The `/obstacle_setup` wire format: `id:x,y,facing` joined by `|`, metres."""
    return '|'.join(f'{i + 1}:{x},{y},{facing}' for i, (x, y, facing) in enumerate(obstacles))


def publish_full_scene(runner, start_pose):
    """Drive every arena publisher once for `start_pose`, and return the captures.

    Mirrors what `control_loop`'s PLANNING_PATH branch plus one
    NAVIGATING_TO_TARGET tick emit, but with the planner output injected from the
    cache so Hybrid A* runs once per module rather than once per test.
    """
    cfg = load_config('test_obstacles.yaml')
    runner.reset()
    runner.node.setup_callback(String(
        data=setup_string([(o['x'], o['y'], o['facing']) for o in cfg['obstacles']])))

    order, checkpoints, unreachable, occ_map = planned_route(start_pose)
    node = runner.node
    node.visiting_order, node.checkpoints = list(order), list(checkpoints)
    node.unreachable, node.occ_map = list(unreachable), occ_map
    node.leg_paths = [planned_leg0(start_pose)] + [None] * (len(order) - 1)
    node.current_target_idx = 0

    node._publish_occupancy_grid()      # also publishes /grid_markers
    node._publish_obstacle_markers()
    node._publish_checkpoint_markers()
    node._publish_current_path()        # also publishes /planned_path
    node._publish_search_progress([(x, y) for x, y, _ in node.leg_paths[0]])
    return runner.captured


# --- Geometry extraction: every numeric field, `frame_id` excluded -----------

def _marker_geometry(marker):
    return (marker.ns, marker.id, marker.type, marker.action, marker.text,
            (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z),
            (marker.pose.orientation.x, marker.pose.orientation.y,
             marker.pose.orientation.z, marker.pose.orientation.w),
            (marker.scale.x, marker.scale.y, marker.scale.z),
            (marker.color.r, marker.color.g, marker.color.b, marker.color.a),
            [(p.x, p.y, p.z) for p in marker.points])


def message_geometry(msg):
    """Canonical geometry of one published message, with `frame_id` left out."""
    if hasattr(msg, 'markers'):
        return [_marker_geometry(m) for m in msg.markers]
    if hasattr(msg, 'info'):        # nav_msgs/OccupancyGrid
        origin = msg.info.origin
        return (msg.info.resolution, msg.info.width, msg.info.height,
                (origin.position.x, origin.position.y, origin.position.z),
                (origin.orientation.x, origin.orientation.y,
                 origin.orientation.z, origin.orientation.w),
                list(msg.data))
    if hasattr(msg, 'poses'):       # nav_msgs/Path
        return [(p.pose.position.x, p.pose.position.y, p.pose.position.z,
                 p.pose.orientation.x, p.pose.orientation.y,
                 p.pose.orientation.z, p.pose.orientation.w) for p in msg.poses]
    raise TypeError(f'no geometry extractor for {type(msg)}')


def scene_geometry_digests(captured):
    return {topic: hashlib.sha256(repr(message_geometry(msgs[-1])).encode()).hexdigest()
            for topic, msgs in captured.items()
            if topic in ('occupancy_grid', 'grid_markers', 'obstacle_markers',
                         'checkpoint_markers', 'planned_path', 'path_markers',
                         'search_progress')}


def arena_frames(captured):
    """The set of `frame_id`s the arena publishers stamped."""
    frames = set()
    for topic, msgs in captured.items():
        if topic in ('cmd_vel', 'bluetooth_tx'):
            continue
        for msg in msgs:
            if hasattr(msg, 'markers'):
                frames.update(m.header.frame_id for m in msg.markers)
            else:
                frames.add(msg.header.frame_id)
    return frames


# ==========================================================================
# 3.1 - identity start pose renders exactly as it does today
# ==========================================================================

def test_identity_start_pose_geometry_values_are_unchanged(runner):
    """Every published geometry value for the `NOT C` start pose is the recorded one.

    With `T_map_odom` the identity, arena coordinates and `odom` coordinates are
    the same numbers, so a fix that only reframes must leave every field here
    untouched. Digests cover all of them; the explicit assertions below name the
    ones a human would want to see in a failure.
    """
    captured = publish_full_scene(runner, IDENTITY_START_POSE)
    digests = scene_geometry_digests(captured)
    assert digests == BASELINE_SCENE_DIGESTS

    grid = runner.one('occupancy_grid')
    assert (grid.info.width, grid.info.height) == BASELINE_GRID_SHAPE
    assert grid.info.resolution == 0.1
    assert (grid.info.origin.position.x, grid.info.origin.position.y) == (-1.0, -1.0)
    assert grid.info.origin.orientation.w == 1.0
    assert sum(v == 100 for v in grid.data) == BASELINE_GRID_OCCUPIED_CELLS

    grid_lines = {m.ns: m for m in runner.one('grid_markers').markers}
    assert list(grid_lines) == ['grid_lines', 'placement_zone_outline', 'start_box_outline']
    assert len(grid_lines['grid_lines'].points) == 164
    assert [(p.x, p.y) for p in grid_lines['placement_zone_outline'].points] == [
        (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
    assert [(p.x, p.y) for p in grid_lines['start_box_outline'].points] == [
        (0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4), (0.0, 0.0)]

    cubes = [m for m in runner.one('obstacle_markers').markers if m.ns == 'obstacles']
    assert [(m.pose.position.x, m.pose.position.y) for m in cubes] == [
        (0.5, 1.0), (1.2, 0.4), (1.7, 1.5), (0.9, 1.8), (1.5, 0.9)]

    arrows = [m for m in runner.one('checkpoint_markers').markers if m.ns == 'checkpoints']
    assert [(m.pose.position.x, m.pose.position.y) for m in arrows] == \
        [(x, y) for x, y, _ in BASELINE_CHECKPOINTS]

    path = runner.one('planned_path')
    assert [(p.pose.position.x, p.pose.position.y) for p in path.poses] == \
        [(x, y) for x, y, _ in planned_leg0(IDENTITY_START_POSE)]


def test_arena_publishers_agree_on_a_single_frame(runner):
    """All arena data shares one `frame_id`, whatever that frame is called.

    The fix relabels these together or not at all - a half-applied
    `arena_frame` parameter would split the scene across two frames and draw
    part of the arena in the wrong place. Deliberately does not assert the
    frame's NAME: pre-fix it is `'odom'`, post-fix `'map'`, and for the identity
    start pose those two coincide numerically, which is exactly requirement 3.1.
    """
    captured = publish_full_scene(runner, IDENTITY_START_POSE)
    assert len(arena_frames(captured)) == 1, arena_frames(captured)


def test_identity_transform_leaves_arena_coordinates_alone():
    """For the identity start pose, arena and `odom` coordinates are identical.

    The premise the test above rests on: `T_map_odom` derived from `(0, 0, 0)` is
    the identity, so relabelling the frame cannot move a single number.
    """
    t_map_odom = IDENTITY_START_POSE
    for point in [(0.0, 0.0, 0.0), (1.0, 1.0, math.pi / 4), (2.0, 0.5, -math.pi)]:
        assert POSE_TRANSFORM.compose(t_map_odom, point) == pytest.approx(point, abs=1e-12)


# ==========================================================================
# 3.6 - /obstacle_setup interpretation
# ==========================================================================

def test_obstacle_setup_is_read_as_arena_metres(runner):
    """`id:x,y,facing` in metres, cube centres exactly on the given coordinates.

    Also pins the label format (`#N [facing] (x, y)` at one decimal, matching the
    arena's 10 cm grid) and that a new set replaces the old one except during a run.
    """
    runner.reset()
    obstacles = [(0.5, 1.0, 'S'), (1.2, 0.4, 'W'), (1.76, 1.55, 'N')]
    runner.node.setup_callback(String(data=setup_string(obstacles)))
    assert runner.node.obstacles == obstacles
    assert runner.node.state is runner.module.State.PLANNING_PATH

    runner.node._publish_obstacle_markers()
    markers = runner.one('obstacle_markers').markers
    cubes = [m for m in markers if m.ns == 'obstacles']
    labels = [m for m in markers if m.ns == 'obstacle_labels']
    assert [(m.pose.position.x, m.pose.position.y) for m in cubes] == \
        [(x, y) for x, y, _ in obstacles]
    assert [m.text for m in labels] == ['#1 [S] (0.5, 1.0)', '#2 [W] (1.2, 0.4)',
                                        '#3 [N] (1.8, 1.6)']

    # A second set replaces the first (the tablet resends after DONE), and
    # abandons any planner still running for the old one.
    gen = runner.node._plan_gen
    runner.node.setup_callback(String(data=setup_string([(0.1, 0.1, 'E')])))
    assert runner.node.obstacles == [(0.1, 0.1, 'E')]
    assert runner.node._plan_gen == gen + 1
    assert runner.node.plan_state == 'PLANNING'

    # ...but never during a run.
    runner.node.state = runner.module.State.NAVIGATING_TO_TARGET
    runner.node.setup_callback(String(data=setup_string([(1.0, 1.0, 'N')])))
    assert runner.node.obstacles == [(0.1, 0.1, 'E')]


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(points=st.lists(
    st.tuples(st.floats(min_value=f32(0.05), max_value=f32(ARENA_SIZE_M - 0.05),
                        allow_nan=False, width=32),
              st.floats(min_value=f32(0.05), max_value=f32(ARENA_SIZE_M - 0.05),
                        allow_nan=False, width=32),
              st.sampled_from(['N', 'S', 'E', 'W'])),
    min_size=1, max_size=6))
def test_published_marker_geometry_equals_the_arena_input(runner, points):
    """For the identity start pose, published geometry equals the arena input.

    The preservation property in its general form: whatever arena coordinates go
    in, the same numbers come out, with no scale, offset or rotation applied
    anywhere in the publish path. `T_map_odom` being the identity for this start
    pose is what makes that a preservation statement rather than a fix-checking
    one.
    """
    runner.reset()
    runner.node.setup_callback(String(data=setup_string(points)))
    runner.node._publish_obstacle_markers()

    markers = runner.one('obstacle_markers').markers
    cubes = [m for m in markers if m.ns == 'obstacles']
    labels = [m for m in markers if m.ns == 'obstacle_labels']
    assert len(cubes) == len(points)

    for (x, y, _facing), cube, label in zip(points, cubes, labels):
        assert (cube.pose.position.x, cube.pose.position.y) == (x, y)
        assert cube.pose.position.z == 0.05
        assert (cube.pose.orientation.z, cube.pose.orientation.w) == (0.0, 1.0)
        assert (label.pose.position.x, label.pose.position.y) == (x, y)


# ==========================================================================
# Rigid-transform invariants (round trip, distances, bearings)
# ==========================================================================

pose_component = st.floats(min_value=f32(-ARENA_SIZE_M), max_value=f32(ARENA_SIZE_M),
                           allow_nan=False, width=32)
yaw_component = st.floats(min_value=f32(-math.pi), max_value=f32(math.pi),
                          allow_nan=False, width=32)


@settings(max_examples=300, deadline=None)
@given(sx=pose_component, sy=pose_component, syaw=yaw_component,
       px=pose_component, py=pose_component, pyaw=yaw_component)
def test_arena_odom_round_trip_returns_the_original_pose(sx, sy, syaw, px, py, pyaw):
    """Into the arena frame and back is the identity, for any start pose.

    Guards the one way a frame fix can silently corrupt data: a transform that
    is not its own inverse. Yaw is compared through the `+/-pi` wrap.
    """
    t_map_odom = (sx, sy, syaw)
    odom_pose = (px, py, pyaw)

    in_arena = POSE_TRANSFORM.compose(t_map_odom, odom_pose)
    back = POSE_TRANSFORM.compose(POSE_TRANSFORM.invert(t_map_odom), in_arena)

    assert back[0] == pytest.approx(odom_pose[0], abs=1e-9)
    assert back[1] == pytest.approx(odom_pose[1], abs=1e-9)
    assert yaw_error(back[2], odom_pose[2]) < 1e-9


@settings(max_examples=300, deadline=None)
@given(sx=pose_component, sy=pose_component, syaw=yaw_component,
       ax=pose_component, ay=pose_component,
       bx=pose_component, by=pose_component)
def test_arena_transform_is_rigid(sx, sy, syaw, ax, ay, bx, by):
    """Distances and relative bearings survive the transform - no scale, no shear.

    If this held only for distance, a reflection would slip through; the bearing
    check pins orientation as well. Together they are what makes "the arena is
    only relabelled, never redrawn" checkable.
    """
    t_map_odom = (sx, sy, syaw)
    a = POSE_TRANSFORM.compose(t_map_odom, (ax, ay, 0.0))
    b = POSE_TRANSFORM.compose(t_map_odom, (bx, by, 0.0))

    assert math.hypot(b[0] - a[0], b[1] - a[1]) == \
        pytest.approx(math.hypot(bx - ax, by - ay), abs=1e-9)

    bearing_before = math.atan2(by - ay, bx - ax)
    bearing_after = math.atan2(b[1] - a[1], b[0] - a[0])
    if math.hypot(bx - ax, by - ay) > 1e-6:
        assert yaw_error(bearing_after, bearing_before + syaw) < 1e-6


# ==========================================================================
# 3.3 - planner numerics, and mdp_algorithm's freedom from frames
# ==========================================================================

def test_planner_numerics_for_the_test_layout_are_unchanged():
    """Visiting order, checkpoints, unreachable list and grid, as recorded.

    Checked for both start poses: the route is start-pose dependent input to the
    planner, so pinning only one of them would leave half the surface untested.
    """
    for start_pose in (IDENTITY_START_POSE, REPORTED_START_POSE):
        order, checkpoints, unreachable, occ_map = planned_route(start_pose)
        assert list(order) == BASELINE_VISITING_ORDER, start_pose
        assert list(unreachable) == BASELINE_UNREACHABLE, start_pose
        assert [tuple(round(float(v), 12) for v in cp) for cp in checkpoints] == \
            BASELINE_CHECKPOINTS, start_pose
        assert occ_map.occupancy_grid.shape == BASELINE_GRID_SHAPE
        assert int(occ_map.occupancy_grid.sum()) == BASELINE_GRID_OCCUPIED_CELLS
        assert grid_digest(occ_map.occupancy_grid) == BASELINE_GRID_SHA256


def test_leg0_geometry_is_unchanged():
    """The dense Hybrid A* leg itself, for both start poses.

    The planner's most expensive and most fragile output. Both calls are cached
    at module scope and shared with the scene tests above.
    """
    for start_pose, (length, first) in BASELINE_LEG0.items():
        leg = planned_leg0(start_pose)
        assert leg, f'no path found from start pose {start_pose}'
        assert len(leg) == length, start_pose
        assert tuple(round(float(v), 12) for v in leg[0]) == first, start_pose
        # Every leg ends within the search's goal tolerance of checkpoint 0,
        # which is itself start-pose independent (see the test above).
        cx, cy, _ = BASELINE_CHECKPOINTS[0]
        assert math.hypot(float(leg[-1][0]) - cx, float(leg[-1][1]) - cy) < 0.05, start_pose


def test_mdp_algorithm_has_no_frame_awareness():
    """`mdp_algorithm` stays in arena centimetres, with no TF concept at all.

    The fix adds frames to `mdp_bringup` only. This reads every string constant
    and import in the package: a frame name as a literal, or a `tf2` import,
    would mean the planner had started caring which frame it is in.

    `base_link` is not in scope: `pure_pursuit_follower`'s standalone node stamps
    its own body-frame twist with it, which is a message header, not arena
    awareness - and `map` as a local variable name (an `OccupancyMap` argument
    all over `hamiltonian.py`) is not a string literal, so it does not trip this.
    """
    offenders = []
    for source in sorted(MDP_ALGORITHM_ROOT.rglob('*.py')):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value in TF_FRAME_NAMES:
                offenders.append(f'{source.name}:{node.lineno} literal {node.value!r}')
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            for name in names:
                if name.split('.')[0] in ('tf2_ros', 'tf2_geometry_msgs', 'tf2_py'):
                    offenders.append(f'{source.name}:{node.lineno} imports {name}')
    assert not offenders, 'mdp_algorithm gained frame awareness: ' + '; '.join(offenders)


# ==========================================================================
# 3.7 - vision, Bluetooth and the `go` gate
# ==========================================================================

def test_yolo_result_handling_is_unchanged(runner):
    """`/yolo_result` is read only while paused to scan, and only the first one.

    Recorded per state: `PAUSE_FOR_SCAN` accepts and strips whitespace, every
    other state ignores the message, and a second detection never overwrites the
    first (the runner reports what it saw on arrival, not the latest frame).
    """
    for state in runner.module.State:
        runner.reset()
        runner.node.state = state
        runner.node.yolo_callback(String(data='  38_AlphabetV  '))
        expected = '38_AlphabetV' if state is runner.module.State.PAUSE_FOR_SCAN else None
        assert runner.node.detected_target_id == expected, state

    runner.node.state = runner.module.State.PAUSE_FOR_SCAN
    runner.node.detected_target_id = '38_AlphabetV'
    runner.node.yolo_callback(String(data='99_Bullseye'))
    assert runner.node.detected_target_id == '38_AlphabetV'


def test_start_run_gating(runner):
    """`/start_run` accepts only from WAITING_FOR_GO with the plan DONE and the
    pose reset, and says why otherwise."""
    State = runner.module.State
    expected = {
        'WAITING_FOR_SETUP': (False, 'Not ready: no obstacle setup received yet.'),
        'PLANNING_PATH': (False, 'Not ready: still planning the path.'),
        'NAVIGATING_TO_TARGET': (False, 'Ignored: run already in progress (NAVIGATING_TO_TARGET).'),
        'PAUSE_FOR_SCAN': (False, 'Ignored: run already in progress (PAUSE_FOR_SCAN).'),
        'FINISHED': (False, 'Not ready: run finished - put the car back and reset (pixi run reset).'),
        'STOPPED': (False, 'Ignored: stopped - reset the pose first (pixi run reset).'),
    }
    for state in State:
        if state is State.WAITING_FOR_GO:
            continue
        runner.reset()
        runner.node.state = state
        response = runner.node.start_run_callback(Trigger.Request(), Trigger.Response())
        assert (response.success, response.message) == expected[state.name], state
        assert runner.node.state is state
    assert set(expected) | {'WAITING_FOR_GO'} == {s.name for s in State}

    # WAITING_FOR_GO: needs BOTH indicators DONE.
    for plan, reset_done, ok, message in (
            ('PLANNING', True, False, 'Not ready: still planning the path.'),
            ('DONE', False, False, 'Not ready: reset the pose first (pixi run reset).'),
            ('DONE', True, True, 'Run started - navigating to targets.')):
        runner.reset()
        runner.node.state = State.WAITING_FOR_GO
        runner.node.plan_state = plan
        runner.node.reset_done = reset_done
        response = runner.node.start_run_callback(Trigger.Request(), Trigger.Response())
        assert (response.success, response.message) == (ok, message)
        assert runner.node.state is (State.NAVIGATING_TO_TARGET if ok else State.WAITING_FOR_GO)
        # Leaving the start invalidates the reset - a rerun needs a new one.
        assert runner.node.reset_done is (False if ok else reset_done)


def bt_lines(runner, prefix):
    return [m.data for m in runner.captured.get('bluetooth_tx', []) if m.data.startswith(prefix)]


def test_target_line_uses_the_tablets_own_obstacle_number(runner):
    """`TARGET,<tablet's n>,<id|UNKNOWN>` on leaving a scan pause.

    The number is the one the tablet sent in its OBSTACLE line, not the 1-based
    position in our list - they differ as soon as the tablet numbers from 0 or
    skips a number.
    """
    for detected, expected in (('20', 'TARGET,7,20'), (None, 'TARGET,7,UNKNOWN')):
        runner.reset()
        runner.node.tablet_ids = ['3', '7']
        runner.node.visiting_order = [1, 0]      # first stop is our obstacle index 1 -> tablet's 7
        runner.node.state = runner.module.State.PAUSE_FOR_SCAN
        runner.node.detected_target_id = detected
        # Past the 0.6 s scan window, so the pause resolves on this tick.
        runner.node.state_start_time = runner.node.get_now_sec() - 5.0
        runner.node.control_loop()
        assert bt_lines(runner, 'TARGET') == [expected]
        assert runner.node.current_target_idx == 1


def test_robot_line_is_a_tablet_cell_and_compass_letter():
    """`ROBOT,<x>,<y>,<N/E/S/W>` - the 10cm cell (0-19) containing the tracked point.
    Sent by the always-on robot_pose_feedback node, not the runner."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'robot_pose_feedback.py'
    spec = importlib.util.spec_from_file_location('robot_pose_feedback', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    line = mod.robot_cell_line
    assert line(0.15, 0.15, math.pi / 2) == 'ROBOT,1,1,N'      # default start pose
    assert line(0.35, 0.35, math.pi / 2) == 'ROBOT,3,3,N'
    assert line(1.00, 1.00, 0.0) == 'ROBOT,10,10,E'
    assert line(1.00, 1.00, math.pi) == 'ROBOT,10,10,W'
    assert line(1.00, 1.00, -math.pi / 2) == 'ROBOT,10,10,S'
    assert line(1.00, 1.00, math.radians(80)) == 'ROBOT,10,10,N'  # nearest compass point
    assert line(-0.5, 5.0, 0.0) == 'ROBOT,0,19,E'              # clamped to the grid


def test_indicator_lines_are_sent_on_change_only(runner):
    """PLAN/RESET/STATUS go out when they change, not every 50 ms tick."""
    runner.reset()
    runner.node.state = runner.module.State.WAITING_FOR_GO
    runner.node.control_loop()
    runner.node.control_loop()
    assert bt_lines(runner, 'PLAN') == ['PLAN:WAITING']
    assert bt_lines(runner, 'RESET') == ['RESET:WAITING']
    assert bt_lines(runner, 'STATUS') == ['STATUS:Waiting']

    runner.node.plan_state = 'DONE'
    runner.node.reset_done = True
    runner.node.control_loop()
    assert bt_lines(runner, 'STATUS')[-1] == 'STATUS:Ready'
    assert bt_lines(runner, 'PLAN')[-1] == 'PLAN:DONE'
    assert bt_lines(runner, 'RESET')[-1] == 'RESET:DONE'


def test_stop_halts_everything_and_reset_restores_ready(runner):
    """STOP: zeros, planner abandoned, manual drive ignored. reset: same plan, ready again."""
    State = runner.module.State
    runner.reset()
    runner.node.state = State.NAVIGATING_TO_TARGET
    runner.node.plan_state = 'DONE'
    gen = runner.node._plan_gen
    response = runner.node.stop_run_callback(Trigger.Request(), Trigger.Response())
    assert response.success and runner.node.state is State.STOPPED
    assert runner.node._plan_gen == gen + 1 and not runner.node.follower.active
    runner.node.control_loop()
    assert runner.captured['cmd_vel'][-1].twist.linear.x == 0.0
    assert bt_lines(runner, 'STATUS')[-1] == 'STATUS:Stopped'

    runner.node.manual_drive_callback(String(data='f'))
    assert runner.node._manual_until == 0.0            # ignored while stopped

    runner.node.reset_run_callback(Trigger.Request(), Trigger.Response())
    assert runner.node.state is State.WAITING_FOR_GO and runner.node.plan_state == 'DONE'
    assert not runner.node.stopped
    runner.node.manual_drive_callback(String(data='f'))
    assert runner.node._manual_until > 0.0             # accepted again once reset

    # A stop that interrupts planning drops the plan and reset replans.
    runner.reset()
    runner.node.obstacles = [(0.5, 1.0, 'S')]
    runner.node.plan_state = 'PLANNING'
    runner.node.state = State.WAITING_FOR_GO
    runner.node.stop_run_callback(Trigger.Request(), Trigger.Response())
    assert runner.node.plan_state == 'WAITING'
    runner.node.reset_run_callback(Trigger.Request(), Trigger.Response())
    assert runner.node.state is State.PLANNING_PATH and runner.node.plan_state == 'PLANNING'

    # reset refuses mid-run.
    runner.reset()
    runner.node.state = State.NAVIGATING_TO_TARGET
    response = runner.node.reset_run_callback(Trigger.Request(), Trigger.Response())
    assert not response.success


def test_manual_drive_burst_directions(runner):
    """Forward/back and left/right map to the right cmd_vel signs (incl. reversing)."""
    expect = {'f': (1, 0), 'b': (-1, 0), 'fl': (1, 1), 'fr': (1, -1), 'bl': (-1, -1), 'br': (-1, 1)}
    for key, (vs, ws) in expect.items():
        runner.reset()
        runner.node.manual_drive_callback(String(data=key))
        v, w = runner.node._manual_cmd
        assert (v > 0) - (v < 0) == vs and (w > 0) - (w < 0) == ws, key
    runner.reset()
    runner.node.state = runner.module.State.PAUSE_FOR_SCAN
    runner.node.manual_drive_callback(String(data='f'))
    assert runner.node._manual_until == 0.0            # ignored during a run



# ==========================================================================
# 3.5 - hardware EKF config untouched
# ==========================================================================

def test_ekf_yaml_is_byte_identical_to_its_pre_fix_content():
    """`config/ekf.yaml` is unmodified: the sim overlay must be a separate file.

    Hardware localization is out of scope for this fix - `world_frame` stays
    `odom` and `publish_tf` keeps the EKF as the sole owner of
    `odom -> base_link`. Task 6.6 adds `ekf_sim.yaml` alongside, never on top.
    """
    content = (CONFIG_DIR / 'ekf.yaml').read_bytes()
    assert len(content) == EKF_YAML_BYTES
    assert hashlib.sha256(content).hexdigest() == EKF_YAML_SHA256


def test_no_sim_overlay_has_been_folded_into_ekf_yaml():
    """The values that make hardware localization work, read as values.

    Byte-identity above catches any edit; this says which edits would matter, so
    a future intentional change to `ekf.yaml` fails with a diagnosis instead of
    just a hash mismatch.
    """
    params = load_config('ekf.yaml')['ekf_filter_node']['ros__parameters']
    assert params['world_frame'] == 'odom'
    assert params['odom_frame'] == 'odom'
    assert params['base_link_frame'] == 'base_link'
    assert params['publish_tf'] is True
    assert params['use_sim_time'] is False
    assert params['imu0'] == '/imu/data'

    # Task 6.6's sim overlay, if it exists yet, is a DIFFERENT file - the whole
    # point of the separate-file decision is that this one stays provably unread
    # by the sim.
    overlay = CONFIG_DIR / 'ekf_sim.yaml'
    if overlay.exists():
        assert overlay.read_bytes() != (CONFIG_DIR / 'ekf.yaml').read_bytes()
