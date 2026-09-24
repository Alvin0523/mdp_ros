#!/usr/bin/env python3
"""
Task 1: Automatic Exploration & Image Recognition Runner Node.
Handles 2.0m x 2.0m arena exploration, collision-aware Hybrid A* Ackermann
path execution, YOLO26 image recognition pause & capture, Android Tablet
Bluetooth updates, and auto-stop.

UPDATED 2026-09-03: previously used mdp_algorithm.reeds_shepp_planner
(no collision/obstacle awareness at all - pure Dubins point-to-point) only
for standoff-pose calc + TSP ordering, and NAVIGATING_TO_TARGET never
actually tracked a path - it just published a fixed forward speed for a
hardcoded 2.5s per target regardless of where the target actually was. Both
replaced: mdp_algorithm.collision_aware_planner (ported Hybrid A* + occupancy
grid + reachability-filtered TSP from the teammate's mdp_algo package,
minus its open-loop discrete-command output layer - this project tracks
paths continuously in closed loop instead, see that module's docstring)
now produces the route, and PurePursuitController (mdp_algorithm.
pure_pursuit_follower) actually drives it using this node's own
/odometry/filtered subscription - not a second competing /cmd_vel
publisher node, see that module's docstring for why.
"""

import math
import threading
import traceback

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from enum import Enum, auto
from geometry_msgs.msg import TwistStamped, PoseStamped, Point, PoseWithCovarianceStamped
import tf2_ros
# Imported for its side effect: registering the geometry_msgs type conversions
# tf2 dispatches on. `do_transform_pose` is called through the module below.
import tf2_geometry_msgs
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from mdp_algorithm.planning.collision_aware_planner import plan_leg, plan_visiting_order
from mdp_algorithm.planning.occupancy_map import (
    ARENA_SIZE_CM, CELL_SIZE_CM, GRID_MARGIN_CM, OBSTACLE_SIZE_CM,
    Obstacle, OccupancyMap,
)
from mdp_algorithm.control.pure_pursuit_follower import PurePursuitController, yaw_from_quaternion

# Task 1's physical camera is mounted facing the car's LEFT side, not
# forward - confirmed by the user. Modeled in mini_akm_robot.urdf's
# camera_joint via the CAMERA_YAW_RAD launch-time substitution (task1_sim
# uses +pi/2 = left; task2_sim/bare sim use 0.0 = forward, matching
# task2_runner.py). This constant is what
# hamiltonian.obstacle_to_checkpoint()'s theta_offset corrects for: the
# checkpoint's body heading is chosen so the CAMERA (not the front
# bumper) ends up pointed at the obstacle's face.
#
# DERIVATION (re-derived from scratch 2026-09-17, replacing an earlier
# -pi/2 value that was picked from a single Foxglove eyeball check, not
# derived): obstacle_to_checkpoint() sets body_theta =
# facing_rad + pi - theta_offset. A camera fixed at local body-frame offset
# phi has world bearing body_theta + phi. Requiring that to equal the
# straight-at-the-obstacle bearing (facing_rad + pi) and solving gives
# phi = theta_offset exactly - the camera's local mounting offset and this
# constant must be the same value, not opposites. Per REP-103
# (positive yaw = left), a left-mounted camera is phi = +pi/2, so
# theta_offset must also be +pi/2, not -pi/2. Re-verify against Foxglove
# (checkpoint arrow direction vs. which physical side the camera is on)
# before trusting this for a real run.
TASK1_CAMERA_THETA_OFFSET_RAD = math.pi / 2.0

# Physical start box the car may be placed anywhere within (per direct user
# description) - 40x40cm, at the arena's own (0,0) corner. Visualization
# only (see _publish_grid_lines' start_box marker) - does NOT feed the
# (0.15, 0.15) start POSE used for planning below, which is a separate,
# still-unverified placeholder for exactly where inside this box the car is
# assumed to start.
START_BOX_SIZE_CM = 40.0

# --- Tablet protocol conventions (see mdp_bridge's bluetooth_bridge_node) ---
# The tablet's grid is 20x20 cells of 10cm, origin bottom-left (cell 0,0). ROBOT
# lines report the cell (0-19) that CONTAINS the robot's tracked point, the same
# rule as obstacles: cell = floor(cm / 10). So the default start pose
# (15 cm, 15 cm) is cell (1,1), and a point at (35 cm, 35 cm) is cell (3,3).
# Changed 2026-09-24 from "bottom-left cell of a 20cm footprint" (start was 0,0)
# - tell the Android side. Set ROBOT_FOOTPRINT_CM back to 20.0 and
# TABLET_MAX_CELL to 18 to restore the old meaning.
TABLET_CELL_CM = 10.0
TABLET_MAX_CELL = 19
ROBOT_FOOTPRINT_CM = 0.0

# Manual drive (f/b/fl/fr/bl/br): a fixed-speed burst, steered by a fixed
# wheel angle. yaw rate follows the bicycle model w = v*tan(delta)/L so the
# sign is right for reversing too (steer left + reverse turns the body right).
MANUAL_SPEED_MPS = 0.15
MANUAL_STEER_DEG = 25.0
MANUAL_BURST_S = 0.3
WHEELBASE_M = 0.1433
MANUAL_COMMANDS = {
    'f': (1.0, 0.0), 'b': (-1.0, 0.0),
    'fl': (1.0, 1.0), 'fr': (1.0, -1.0),
    'bl': (-1.0, 1.0), 'br': (-1.0, -1.0),
}

# The EKF's world frame. map -> odom is a static start-pose transform, so the
# start pose in `odom` is always the identity.
ODOM_FRAME = 'odom'
RESET_POS_TOL_M = 0.03
RESET_YAW_TOL_RAD = math.radians(5.0)
RESET_CONFIRM_TIMEOUT_S = 3.0


class State(Enum):
    WAITING_FOR_SETUP = auto()
    PLANNING_PATH = auto()
    # After planning completes the car does NOT drive automatically - it holds
    # here until the `go` trigger (the /start_run service) fires, mirroring the
    # real competition where the tablet sends a separate "start" command once
    # the supervisor is ready. Only then -> NAVIGATING_TO_TARGET.
    WAITING_FOR_GO = auto()
    NAVIGATING_TO_TARGET = auto()
    PAUSE_FOR_SCAN = auto()
    FINISHED = auto()
    # Entered by /stop_run. Zeros are streamed, manual drive is ignored, and
    # the state only changes again on /reset_run (or a new obstacle setup).
    STOPPED = auto()


class Task1Runner(Node):
    def __init__(self):
        super().__init__('task1_runner')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        # Visualization style/topic-name parameters - see
        # mdp_bringup/config/occupancy_grid_viz.yaml for the full set and
        # its scope note (visualization only, not the underlying grid math).
        self._declare_viz_params()

        # Frame every arena-referenced publisher below stamps (see
        # _declare_viz_params for the convention). Read once - it is fixed for
        # the life of the node.
        self.arena_frame = self.get_parameter('arena_frame').value

        # Where the robot starts in the arena frame. Supplied by the launch file
        # that also broadcasts map -> odom from the same numbers (sim: the Gazebo
        # spawn pose; hardware: the placement in the 40x40cm start box), so the
        # planned route starts where the car actually is. The defaults match the
        # documented placement for a bare `ros2 run`.
        self.declare_parameter('start_x', 0.15)
        self.declare_parameter('start_y', 0.15)
        self.declare_parameter('start_yaw', math.pi / 2.0)

        # Publishers & Subscribers
        # /cmd_vel is the correct target: real.launch.py and sim.launch.py both
        # remap ackermann_steering_controller's actual reference subscription
        # to /cmd_vel, so publishing directly to
        # /ackermann_steering_controller/reference (as this used to do) never
        # reached the controller - dead traffic to an unsubscribed topic.
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bt_pub = self.create_publisher(String, '/bluetooth_tx', 10)

        # Visualization only - nothing here feeds control. Lets the planned
        # grid/obstacles/path actually be seen in Foxglove instead of only
        # existing as numbers in log lines. Split into one topic per marker
        # type (previously all crammed onto one /obstacle_markers topic,
        # including the path line-strip - confirmed confusing when read
        # back live: "why is there a path in the obstacle marker topic").
        #
        # TRANSIENT_LOCAL on all three: confirmed by direct testing that a
        # plain volatile publisher means any Foxglove session connecting
        # even slightly after a one-shot publish (grid/obstacles/grid-lines
        # are each published once, not on every tick) never sees it at all
        # (ros2 topic echo --once just hangs). Durability fixes this - a
        # late subscriber gets the last message on connect, same as how map
        # servers normally publish nav_msgs/OccupancyGrid.
        grid_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(
            OccupancyGrid, self.get_parameter('occupancy_grid_topic').value, grid_qos)
        self.path_pub = self.create_publisher(
            Path, self.get_parameter('planned_path_topic').value, 10)
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter('obstacle_markers_topic').value, grid_qos)
        self.grid_marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter('grid_line_markers_topic').value, grid_qos)
        self.checkpoint_marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter('checkpoint_markers_topic').value, grid_qos)
        # Path markers DO change every tick while navigating (new leg,
        # moving playhead) - plain volatile QoS is fine/correct here, unlike
        # the other two which are one-shot.
        self.path_marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter('path_markers_topic').value, 10)
        # Live Hybrid A* search progress - see plan_leg()'s progress_callback
        # and _plan_all_legs_worker() below. Without this, a leg's search is a
        # black box: Foxglove shows nothing at all for however long it
        # takes, with no way to tell "still working" from "hung." Volatile
        # QoS is fine - this republishes every progress_interval nodes while
        # a search is running, nothing to catch up on for a late subscriber.
        self.search_progress_pub = self.create_publisher(MarkerArray, '/search_progress', 10)

        self.create_subscription(String, '/obstacle_setup', self.setup_callback, 10)
        self.create_subscription(String, '/yolo_result', self.yolo_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        # Manual drive from the tablet (f/b/fl/fr/bl/br), relayed by
        # bluetooth_bridge_node. Handled here rather than in the bridge because
        # this node already streams zeros on /cmd_vel while idle - a second
        # publisher on that topic would fight it.
        self.create_subscription(String, '/manual_drive', self.manual_drive_callback, 10)
        # robot_localization's ekf_node subscribes to this to reset its state.
        # The pose reset itself is a base service (/reset_pose, robot_pose_feedback.py).
        # The runner only follows the EKF's /set_pose to keep its own state right.
        self.reset_pose_client = self.create_client(Trigger, '/reset_pose')
        self.create_subscription(PoseWithCovarianceStamped, '/set_pose', self.set_pose_callback, 10)

        # `go` trigger: a service (not a topic) so the caller gets a clear
        # accepted/rejected acknowledgement - matches the tablet's one-shot
        # "start now" command. Only accepted while WAITING_FOR_GO (i.e. after
        # setup + planning); rejected with a reason otherwise.
        self.start_run_srv = self.create_service(
            Trigger, '/start_run', self.start_run_callback)
        # Stop the follower and the planner, then hold zeros until /reset_run.
        self.stop_run_srv = self.create_service(
            Trigger, '/stop_run', self.stop_run_callback)
        # Put the pose/odometry back at the start pose. Nothing else: obstacles
        # and the plan are kept so a rerun needs no new setup.
        self.reset_run_srv = self.create_service(
            Trigger, '/reset_run', self.reset_run_callback)

        # TF buffer/listener - same pattern task2_runner uses. Needed because
        # /odometry/filtered reports in `odom` (the dead-reckoning frame, created
        # wherever the robot started) while everything this node plans and
        # publishes is in the arena frame. odom_callback below converts one to the
        # other instead of assuming they coincide, which they only do for a
        # (0, 0, 0) start pose.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Control Loop @ 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)

        # Planner & path-following state
        self.follower = PurePursuitController()
        self.current_pose = (0.0, 0.0, math.pi / 2)  # x, y, yaw - updated by odom_callback
        self.state = State.WAITING_FOR_SETUP

        self.obstacles = []          # (x_m, y_m, facing) as received from /obstacle_setup
        self.tablet_ids = []         # tablet's own obstacle number per entry of self.obstacles
        self.visiting_order = []     # obstacle indices, in visit order (mirrors old contract)
        # One entry per obstacle in visiting_order; None until
        # _plan_all_legs_worker (background thread, see below) fills it in -
        # legs are planned back-to-back in the background, not gated on the
        # robot's physical progress through the route.
        self.leg_paths = []          # one dense path (metres) per obstacle, or None if not planned yet
        self.checkpoints = []        # one (x,y,theta) stand-off pose (metres/rad) per obstacle, same order as visiting_order
        self.unreachable = []
        self.current_target_idx = 0
        self.occ_map = None          # set once planning runs - for visualization only
        # Background planning thread (see _plan_all_legs_worker) - plans
        # every leg back-to-back as soon as visiting order is known, NOT
        # gated on the robot physically reaching each checkpoint first, so
        # leg 2's search can already be running while the robot is still
        # driving/pausing on leg 1. control_loop() just polls
        # self.leg_paths[idx] each tick and starts driving the moment it's
        # filled in - see NAVIGATING_TO_TARGET's handling below.
        self._planning_thread = None
        # Bumped whenever the current plan is abandoned (new setup, stop). The
        # planning thread captures the value it started with and drops its
        # results if it no longer matches - Hybrid A* can't be interrupted
        # mid-search, so a stale thread just finishes into an orphaned list.
        self._plan_gen = 0

        # Tablet indicators (PLAN / RESET / STATUS lines on /bluetooth_tx).
        self.plan_state = 'WAITING'    # WAITING | PLANNING | DONE
        self.reset_done = False        # True once the pose is confirmed at the start
        self.stopped = False           # set by /stop_run, cleared by /reset_run
        self._reset_pending = False
        self._reset_requested_at = 0.0
        self._last_sent = {}           # indicator key -> last line sent, to publish on change only
        self._last_heartbeat = 0.0
        # Tunable live: ros2 param set /task1_runner manual_speed_mps 0.3
        # While True the runner does NOT stream zero /cmd_vel when idle (waiting for
        # go / finished), so another publisher (dist, rotate, circle, teleop) can
        # move the car. STOP and a real run are unaffected.
        # ros2 param set /task1_runner external_control true
        self.declare_parameter('external_control', False)
        self.declare_parameter('manual_speed_mps', MANUAL_SPEED_MPS)
        self.declare_parameter('manual_steer_deg', MANUAL_STEER_DEG)
        self.declare_parameter('manual_burst_s', MANUAL_BURST_S)
        self._manual_cmd = (0.0, 0.0)
        self._manual_until = 0.0
        self._manual_active = False

        self.detected_target_id = None
        self.state_start_time = self.get_now_sec()
        self.get_logger().info("Task 1 Runner Node Initialized! Waiting for obstacle setup...")

    def _declare_viz_params(self):
        """Declares every parameter in occupancy_grid_viz.yaml with the same
        defaults that file documents, so this node runs sensibly even if
        launched without that config (e.g. ros2 run directly, as done for
        local testing) - the YAML overrides these when loaded via launch."""
        # The ARENA frame: origin at the arena's bottom-left corner with axes
        # matching the planner's coordinates, i.e. the frame every number this
        # node publishes has always been expressed in. `map` per REP-105, which
        # is also Foxglove's and RViz's default fixed frame. NOT `odom` - that is
        # the dead-reckoning frame, created wherever the robot happened to start,
        # and stamping arena coordinates with it drew the whole arena rotated and
        # offset by the start pose. The map -> odom edge is broadcast by the
        # launch files (see task1_sim.launch.py / real.launch.py).
        self.declare_parameter('arena_frame', 'map')

        self.declare_parameter('occupancy_grid_topic', '/occupancy_grid')
        self.declare_parameter('obstacle_markers_topic', '/obstacle_markers')
        self.declare_parameter('grid_line_markers_topic', '/grid_markers')
        self.declare_parameter('path_markers_topic', '/path_markers')
        self.declare_parameter('planned_path_topic', '/planned_path')
        self.declare_parameter('checkpoint_markers_topic', '/checkpoint_markers')

        self.declare_parameter('grid_line_width_m', 0.003)
        self.declare_parameter('grid_line_color', [0.3, 0.4, 0.5, 0.5])
        self.declare_parameter('zone_outline_width_m', 0.015)
        self.declare_parameter('zone_outline_color', [0.3, 0.75, 1.0, 0.95])

        self.declare_parameter('obstacle_color', [0.9, 0.5, 0.1, 0.9])
        self.declare_parameter('obstacle_label_color', [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter('obstacle_label_height_m', 0.20)

        self.declare_parameter('path_line_width_m', 0.02)
        self.declare_parameter('path_color', [0.1, 0.6, 1.0, 0.9])

        self.declare_parameter('checkpoint_color', [0.2, 1.0, 0.4, 0.95])
        self.declare_parameter('checkpoint_arrow_length_m', 0.15)

    def get_start_pose(self):
        """The robot's `(x, y, yaw)` start pose in the arena frame, in metres/rad.

        Read from parameters rather than hardcoded, so this node, the Gazebo spawn
        and the static map -> odom transform all quote the same pose - see the
        `start_x`/`start_y`/`start_yaw` declarations in __init__.
        """
        return (float(self.get_parameter('start_x').value),
                float(self.get_parameter('start_y').value),
                float(self.get_parameter('start_yaw').value))

    def get_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def odom_callback(self, msg: Odometry) -> None:
        """Convert the incoming odometry pose into the arena frame, then store it.

        `/odometry/filtered` is an `odom`-frame message (that is its contract, and
        it stays that way - nothing here republishes it). The planned path,
        checkpoints and every marker this node emits are in `self.arena_frame`, so
        the pose handed to the follower has to be converted, not passed through:
        `odom` is created at the robot's start pose with identity orientation, so
        consuming it raw applies a constant rotation of `start_yaw` and a constant
        offset of `(start_x, start_y)` to every tracking error.

        The transform comes from TF rather than from the start-pose parameters so
        there is exactly one authority for it (the launch file's `map -> odom`
        broadcaster) - if a real localization source ever replaces that static
        broadcaster, this code does not change.

        On lookup failure the previous pose is KEPT and the update is dropped.
        Falling back to the raw pose would be worse than stale data: it would
        silently reintroduce the 90 degree error for as long as TF was unavailable,
        intermittently and without a symptom in the logs.
        """
        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.pose = msg.pose.pose

        try:
            tf = self.tf_buffer.lookup_transform(
                self.arena_frame, msg.header.frame_id, msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException) as exc:
            self.get_logger().warn(
                f"No {self.arena_frame} <- {msg.header.frame_id} transform yet "
                f"({exc}); keeping previous pose {self.current_pose}",
                throttle_duration_sec=2.0)
            return

        pose_arena = tf2_geometry_msgs.do_transform_pose(pose_in.pose, tf)
        p = pose_arena.position
        yaw = yaw_from_quaternion(pose_arena.orientation)
        self.current_pose = (p.x, p.y, yaw)
        self.follower.update_pose(p.x, p.y, yaw)

        if self._reset_pending:
            sx, sy, syaw = self.get_start_pose()
            yaw_err = math.atan2(math.sin(yaw - syaw), math.cos(yaw - syaw))
            if (math.hypot(p.x - sx, p.y - sy) < RESET_POS_TOL_M
                    and abs(yaw_err) < RESET_YAW_TOL_RAD):
                self._reset_pending = False
                self.reset_done = True
                self.get_logger().info("Reset confirmed: pose is at the start.")
            elif self.get_now_sec() - self._reset_requested_at > RESET_CONFIRM_TIMEOUT_S:
                self._reset_pending = False
                self.get_logger().warn(
                    f"Reset not confirmed after {RESET_CONFIRM_TIMEOUT_S:.0f}s - pose "
                    f"({p.x:.2f}, {p.y:.2f}, {math.degrees(yaw):.0f}deg) is not at the start "
                    f"pose {self.get_start_pose()}. Is the EKF running / did it take /set_pose?")

    def setup_callback(self, msg: String):
        """`id:x,y,facing|id:x,y,facing|...` (metres). A new set replaces the old
        one - including a finished plan - but never interrupts a run in progress."""
        if self.state in (State.NAVIGATING_TO_TARGET, State.PAUSE_FOR_SCAN):
            self.get_logger().warn("Obstacle setup ignored: a run is in progress. Stop it first.")
            return

        obstacles, ids = [], []
        for item in msg.data.strip().split('|'):
            if ':' not in item:
                continue
            try:
                obs_id, data = item.split(':')
                parts = data.split(',')
                x, y, face = float(parts[0]), float(parts[1]), parts[2].strip().upper()
            except (ValueError, IndexError):
                self.get_logger().warn(f"Skipping malformed obstacle entry {item!r}")
                continue
            if face not in ('N', 'E', 'S', 'W'):
                self.get_logger().warn(f"Skipping obstacle {obs_id!r}: facing {face!r} is not N/E/S/W")
                continue
            obstacles.append((x, y, face))
            ids.append(obs_id.strip())

        if not obstacles:
            self.get_logger().warn("Obstacle setup contained no valid obstacles - ignored.")
            return

        self._plan_gen += 1   # abandon any planner still running for the old set
        self.obstacles = obstacles
        self.tablet_ids = ids
        self.visiting_order = []
        self.checkpoints = []
        self.unreachable = []
        self.leg_paths = []
        self.current_target_idx = 0
        self.follower.set_path([])
        self.plan_state = 'PLANNING'
        self.state = State.PLANNING_PATH
        self.state_start_time = self.get_now_sec()
        cells = ' | '.join(
            f"#{oid} ({int(math.floor(x * 100.0 / CELL_SIZE_CM + 1e-6))},"
            f"{int(math.floor(y * 100.0 / CELL_SIZE_CM + 1e-6))}) {face}"
            for oid, (x, y, face) in zip(ids, obstacles))
        self.get_logger().info(
            f"Loaded {len(self.obstacles)} obstacles (cell x,y, facing): {cells}")

    def _tablet_id(self, obstacle_idx: int) -> str:
        """The tablet's own number for obstacle `obstacle_idx` (0-based index into
        self.obstacles), falling back to idx+1 if ids were never supplied."""
        if obstacle_idx < len(self.tablet_ids):
            return self.tablet_ids[obstacle_idx]
        return str(obstacle_idx + 1)

    def _current_obstacle_label(self) -> str:
        if self.current_target_idx < len(self.visiting_order):
            return self._tablet_id(self.visiting_order[self.current_target_idx])
        return '?'

    def _status_text(self) -> str:
        if self.stopped:
            return 'Stopped'
        if self.state == State.NAVIGATING_TO_TARGET:
            return f'Going to obstacle {self._current_obstacle_label()}'
        if self.state == State.PAUSE_FOR_SCAN:
            return f'Scanning obstacle {self._current_obstacle_label()}'
        if self.state == State.FINISHED:
            return 'Finished'
        if (self.state == State.WAITING_FOR_GO and self.plan_state == 'DONE'
                and self.reset_done):
            return 'Ready'
        return 'Waiting'

    def _sync_indicators(self):
        """PLAN / RESET / STATUS lines, sent only when they change. The bridge
        remembers the latest of each and resends them at link-up."""
        lines = {
            'PLAN': f'PLAN:{self.plan_state}',
            'RESET': f'RESET:{"DONE" if self.reset_done else "WAITING"}',
            'STATUS': f'STATUS:{self._status_text()}',
        }
        for key, line in lines.items():
            if self._last_sent.get(key) != line:
                self._last_sent[key] = line
                self.send_bt(line)

    def yolo_callback(self, msg: String):
        if self.state == State.PAUSE_FOR_SCAN and self.detected_target_id is None:
            self.detected_target_id = msg.data.strip()
            self.get_logger().info(f"YOLO26 Identified Target: {self.detected_target_id}")

    def start_run_callback(self, request, response):
        """`/start_run` (the tablet's BEGIN / `pixi run go`). Starts only from
        WAITING_FOR_GO with the plan DONE and the pose reset; otherwise rejects
        with a reason so the caller knows why nothing happened."""
        if self.state == State.WAITING_FOR_GO:
            if self.plan_state != 'DONE':
                response.success = False
                response.message = "Not ready: still planning the path."
            elif not self.reset_done:
                response.success = False
                response.message = "Not ready: reset the pose first (pixi run reset)."
            else:
                self.state = State.NAVIGATING_TO_TARGET
                self.state_start_time = self.get_now_sec()
                self.current_target_idx = 0
                self._manual_until = 0.0
                # The car is about to leave the start, so it needs a fresh reset
                # before it can run again.
                self.reset_done = False
                response.success = True
                response.message = "Run started - navigating to targets."
                self.get_logger().info("Received `go` - starting navigation.")
        elif self.state == State.WAITING_FOR_SETUP:
            response.success = False
            response.message = "Not ready: no obstacle setup received yet."
        elif self.state == State.PLANNING_PATH:
            response.success = False
            response.message = "Not ready: still planning the path."
        elif self.state == State.STOPPED:
            response.success = False
            response.message = "Ignored: stopped - reset the pose first (pixi run reset)."
        elif self.state == State.FINISHED:
            response.success = False
            response.message = "Not ready: run finished - put the car back and reset (pixi run reset)."
        else:
            response.success = False
            response.message = f"Ignored: run already in progress ({self.state.name})."
        if not response.success:
            self.get_logger().warn(f"`go` rejected: {response.message}")
        self._sync_indicators()
        return response

    def stop_run_callback(self, request, response):
        """`/stop_run` (the tablet's STOP / `pixi run stop`). Halts the follower
        and the planner, and holds zeros on /cmd_vel until /reset_run."""
        self._plan_gen += 1               # abandon any planner still running
        self.stopped = True
        self.reset_done = False           # wherever the car is now, it's not a verified start
        self._reset_pending = False
        self._manual_until = 0.0
        self.follower.set_path([])
        if self.plan_state == 'PLANNING':
            self.plan_state = 'WAITING'   # aborted mid-plan; /reset_run replans from the kept obstacles
        self.state = State.STOPPED
        self.state_start_time = self.get_now_sec()
        self.send_cmd(0.0, 0.0)
        self.get_logger().warn("STOP - halted; holding zeros until reset.")
        self._sync_indicators()
        response.success = True
        response.message = "Stopped."
        return response

    def reset_run_callback(self, request, response):
        """`/reset_run` (tablet reset). Asks the base to put the EKF pose back at
        the start pose (/reset_pose); set_pose_callback then updates the run state.
        Obstacles and the plan are kept."""
        if self.state in (State.NAVIGATING_TO_TARGET, State.PAUSE_FOR_SCAN):
            response.success = False
            response.message = "Ignored: run in progress - stop it first."
            return response
        if not self.reset_pose_client.wait_for_service(timeout_sec=1.0):
            response.success = False
            response.message = "/reset_pose not available (is robot_pose_feedback running?)"
            return response
        self.reset_pose_client.call_async(Trigger.Request())
        response.success = True
        response.message = "Reset requested; RESET turns DONE once the pose is at the start."
        return response

    def set_pose_callback(self, msg: PoseWithCovarianceStamped):
        """The EKF pose was reset (from /reset_run or `pixi run reset`): RESET turns
        DONE only once odometry actually reports the start pose (odom_callback)."""
        if self.state in (State.NAVIGATING_TO_TARGET, State.PAUSE_FOR_SCAN):
            self.get_logger().warn("Pose was reset during a run - stop it first.")
            return

        self.reset_done = False
        self._reset_pending = True
        self._reset_requested_at = self.get_now_sec()
        self.stopped = False
        self._manual_until = 0.0
        self.follower.set_path([])

        if self.state in (State.FINISHED, State.STOPPED):
            self.current_target_idx = 0
            if self.plan_state == 'DONE':
                self.state = State.WAITING_FOR_GO
            elif self.obstacles:
                self._plan_gen += 1
                self.leg_paths = []
                self.plan_state = 'PLANNING'
                self.state = State.PLANNING_PATH
            else:
                self.state = State.WAITING_FOR_SETUP
            self.state_start_time = self.get_now_sec()

        self.get_logger().info("Reset requested - waiting for odometry to report the start pose.")
        self._sync_indicators()

    def manual_drive_callback(self, msg: String):
        """f/b/fl/fr/bl/br from the tablet: a short fixed burst on /cmd_vel,
        then zeros. Ignored during a run and after STOP until /reset_run."""
        key = msg.data.strip().lower()
        if key not in MANUAL_COMMANDS:
            self.get_logger().warn(f"Unknown manual drive command {msg.data!r}")
            return
        if self.stopped or self.state in (State.NAVIGATING_TO_TARGET, State.PAUSE_FOR_SCAN):
            self.get_logger().info(f"Manual drive {key!r} ignored (run in progress or stopped).")
            return
        direction, steer = MANUAL_COMMANDS[key]
        # Clamped so a bad value can't send the car flying: 0.05-0.5 m/s,
        # steering 5-30 deg, burst 0.1-1.0 s.
        speed = min(0.5, max(0.05, float(self.get_parameter('manual_speed_mps').value)))
        steer_deg = min(30.0, max(5.0, float(self.get_parameter('manual_steer_deg').value)))
        burst_s = min(1.0, max(0.1, float(self.get_parameter('manual_burst_s').value)))
        v = direction * speed
        w = v * math.tan(math.radians(steer_deg)) / WHEELBASE_M * steer
        self._manual_cmd = (v, w)
        self._manual_until = self.get_now_sec() + burst_s
        self._manual_active = True
        # Moving the car by hand means it's no longer at a verified start pose.
        self.reset_done = False
        self._reset_pending = False

    def send_cmd(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def send_bt(self, text: str):
        msg = String()
        msg.data = text
        self.bt_pub.publish(msg)

    def control_loop(self):
        now = self.get_now_sec()
        elapsed = now - self.state_start_time

        # Repeat the indicator lines every few seconds. /bluetooth_tx is
        # volatile, so a bridge that comes up after this node would otherwise
        # never learn the current PLAN/RESET/STATUS/ROBOT until one changes. The
        # bridge drops repeats, so the tablet still only sees real changes.
        if now - self._last_heartbeat >= 2.0:
            self._last_heartbeat = now
            self._last_sent.clear()
        self._sync_indicators()

        # Manual drive burst from the tablet (never active during a run - see
        # manual_drive_callback). Ends with an explicit zero, then the normal
        # per-state zero streaming resumes.
        if self._manual_until > now:
            self.send_cmd(*self._manual_cmd)
            return
        if self._manual_active:
            self._manual_active = False
            self.send_cmd(0.0, 0.0)

        # STATE 1: Planning Path after receiving setup
        if self.state == State.PLANNING_PATH:
            # /obstacle_setup gives obstacle position in METRES (matches this
            # node's own start_pose/current_pose convention) - convert to cm,
            # the unit collision_aware_planner/occupancy_map work in. This is
            # now the obstacle's true CONTINUOUS centre, not a grid-floored
            # index - "placed at 1m,1m" means the 10x10cm block is centred
            # exactly there (see occupancy_map.Obstacle's docstring).
            obstacles_grid = [(x_m * 100.0, y_m * 100.0, face) for x_m, y_m, face in self.obstacles]

            # Publish the grid/obstacles first - these only depend on the
            # obstacle list, not on any planning finishing.
            preview_obstacles = [Obstacle(x_cm=x_cm, y_cm=y_cm, facing=face, id=i)
                                  for i, (x_cm, y_cm, face) in enumerate(obstacles_grid)]
            self.occ_map = OccupancyMap(preview_obstacles)
            self._publish_occupancy_grid()
            self._publish_obstacle_markers()

            # From the start_x/start_y/start_yaw parameters the launch file also
            # derives the static map -> odom transform from, so the route starts
            # where the robot actually is (it used to be a third independent
            # literal here, which is how the arena and the robot drifted apart).
            #
            # NOT (0.0, 0.0, ...): confirmed by direct testing that a rear-axle
            # start pose literally at the arena's inside corner is physically
            # unreachable for Hybrid A* to escape - the car's own front bumper
            # is already inside an obstacle's inflation margin on two sides at
            # once if one sits near the corner, and every turn choice's swept
            # front point re-enters that margin before the car can clear it (a
            # real kinematic constraint, not a search bug). No longer an ARENA
            # wall issue since the 2026-09-04 border removal (occupancy_map.py)
            # - only obstacle inflation can still cause this. (0.15, 0.15) is a
            # reasonable placeholder for "rear axle at the inner corner of the
            # marked start box," but is NOT verified against the actual
            # competition start-box convention/size. Confirm against the real
            # starting setup before trusting this for a run.
            start_pose = self.get_start_pose()

            # Only the visiting order + checkpoints (Hamiltonian TSP, cheap -
            # see collision_aware_planner.plan_visiting_order()) are computed
            # here, NOT any leg's dense Hybrid A* path - that starts right
            # below, in the background thread.
            (self.visiting_order, self.checkpoints,
             self.unreachable, self.occ_map) = plan_visiting_order(
                obstacles_grid, start_pose, theta_offset=TASK1_CAMERA_THETA_OFFSET_RAD)
            self._publish_occupancy_grid()
            self._publish_obstacle_markers()
            self._publish_checkpoint_markers()

            # 1-indexed in both log lines below (obstacle_idx + 1), matching
            # /checkpoint_markers' "CP #N" labels and the Bluetooth
            # TARGET,N,... convention (self.visiting_order[i] + 1) - was
            # printing self.visiting_order/self.unreachable raw (0-indexed),
            # inconsistent with everything else that shows an obstacle
            # number, confirmed confusing.
            if self.unreachable:
                self.get_logger().warn(
                    f"Obstacles with no valid scan checkpoint, skipped: {[self._tablet_id(i) for i in self.unreachable]}")
            self.get_logger().info(
                f"Visiting Order Calculated: {[self._tablet_id(i) for i in self.visiting_order]}")

            self.leg_paths = [None] * len(self.visiting_order)
            self.current_target_idx = 0
            # Do NOT drive yet: hold in WAITING_FOR_GO until the `go` trigger
            # (/start_run service). Leg planning still runs in the background
            # NOW, so by the time `go` fires the legs are ready (or nearly) -
            # the car just isn't allowed to move until then.
            self.state = State.WAITING_FOR_GO
            self.state_start_time = now
            self.get_logger().info(
                "Planning complete. Holding - waiting for `go` (/start_run service).")

            # Plans every leg back-to-back, in the background, starting NOW
            # - not gated on the robot physically reaching each checkpoint
            # first (see this thread's own docstring). NAVIGATING_TO_TARGET
            # below picks up leg 0 the moment it's ready, same as before;
            # every later leg just gets a head start instead of only
            # starting once the robot arrives at the checkpoint before it.
            self._planning_thread = threading.Thread(
                target=self._plan_all_legs_worker,
                args=(start_pose, self._plan_gen, self.leg_paths, self.checkpoints, self.occ_map),
                daemon=True)
            self._planning_thread.start()

        # STATE 1b: Planning done, car held until `go` (see start_run_callback).
        elif self.state == State.WAITING_FOR_GO:
            if not self.get_parameter('external_control').value:
                self.send_cmd(0.0, 0.0)

        # STATE 2: Navigating to current target standoff pose - now actually
        # tracks the Hybrid A*-planned path via PurePursuitController,
        # instead of blindly driving forward for a fixed 2.5s.
        elif self.state == State.NAVIGATING_TO_TARGET:
            # Leg may still be mid-search in the background thread (see
            # _plan_all_legs_worker) - check every tick, not just once at
            # the state transition, so the robot starts driving the instant
            # it's ready instead of only when arrival at the PREVIOUS
            # checkpoint happened to trigger a (re)check.
            if self.current_target_idx >= len(self.leg_paths):
                self.send_cmd(0.0, 0.0)
                self.state = State.FINISHED
                self.get_logger().info("Nothing to visit - finished.")
                return

            if not self.follower.active:
                if not self._start_current_leg():
                    self.send_cmd(0.0, 0.0)   # still waiting on the background planner
                    return
                if self.state != State.NAVIGATING_TO_TARGET:
                    return   # _start_current_leg() moved us to PAUSE_FOR_SCAN (empty leg)

            self._publish_current_path()
            cmd = self.follower.compute_cmd()
            if cmd is None or self.follower.is_done():
                self.send_cmd(0.0, 0.0)
                self.detected_target_id = None
                self.state = State.PAUSE_FOR_SCAN
                self.state_start_time = now
                self.get_logger().info(f"Arrived at Target Standoff {self.current_target_idx + 1}. Scanning...")
            else:
                self.send_cmd(*cmd)

        # STATE 3: Pause for YOLO26 scanning & Bluetooth update
        elif self.state == State.PAUSE_FOR_SCAN:
            self.send_cmd(0.0, 0.0)

            if self.detected_target_id is not None or elapsed > 0.6:
                target_id = self.detected_target_id if self.detected_target_id else "UNKNOWN"
                obs_num = self._tablet_id(self.visiting_order[self.current_target_idx])

                self.send_bt(f"TARGET,{obs_num},{target_id}")
                self.get_logger().info(f"Updated Android Tablet: TARGET,{obs_num},{target_id}")

                self.current_target_idx += 1
                if self.current_target_idx >= len(self.visiting_order):
                    self.state = State.FINISHED
                    self.get_logger().info("All targets processed! Auto-stopping...")
                else:
                    self.state = State.NAVIGATING_TO_TARGET
                    self.state_start_time = now
                    # No _start_current_leg() call here - self.follower.active
                    # is already False (compute_cmd() clears it on arrival),
                    # so the very next NAVIGATING_TO_TARGET tick's own check
                    # picks this up automatically (and keeps polling if the
                    # background thread hasn't finished this leg yet).

        # STATE 4: Finished & Auto-Stopped
        elif self.state == State.FINISHED:
            if not self.get_parameter('external_control').value:
                self.send_cmd(0.0, 0.0)

        # STATE 5: Stopped by /stop_run - keep publishing zeros until reset
        elif self.state == State.STOPPED:
            self.send_cmd(0.0, 0.0)

    def _publish_occupancy_grid(self):
        """Publishes the planner's occupancy grid (obstacle inflation only -
        no arena-border wall, see occupancy_map.py's module docstring) once,
        right after planning - the grid itself doesn't change mid-run.
        occ_map.occupancy_grid is indexed [x_g][y_g] (padded-grid convention,
        see occupancy_map.py); nav_msgs/OccupancyGrid's row-major data is
        index = x + y*width, so iterate y outer, x inner. The grid's cell
        (0,0) is GRID_MARGIN_CM into the padding, not the nominal placement
        zone's own corner - info.origin below shifts it back so this renders
        aligned with real-world (obstacle/robot) coordinates.

        REMOVED 2026-09-05: used to also stamp each checkpoint's cell into
        the published grid with a distinct value (grey, in Foxglove's
        default OccupancyGrid colour scheme) - dropped per direct request.
        /checkpoint_markers (exact continuous position, not grid-snapped)
        is the only checkpoint representation now - see that function's
        docstring for why a grid-cell stamp was the less precise of the
        two anyway."""
        if self.occ_map is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.arena_frame
        msg.info.resolution = CELL_SIZE_CM / 100.0
        height, width = self.occ_map.occupancy_grid.shape
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = -GRID_MARGIN_CM / 100.0
        msg.info.origin.position.y = -GRID_MARGIN_CM / 100.0
        msg.info.origin.orientation.w = 1.0

        grid = self.occ_map.occupancy_grid
        msg.data = [int(grid[x][y]) * 100 for y in range(height) for x in range(width)]
        self.grid_pub.publish(msg)
        self._publish_grid_lines()

    def _publish_grid_lines(self):
        """Explicit cell-boundary lines across the 2x2 m arena, not
        just the OccupancyGrid raster's implicit cell colouring - drawn as
        two LINE_LIST markers (horizontal/vertical) so every individual 5cm
        cell is visible in Foxglove's 3D panel, not only where occupancy
        changes. A second, brighter LINE_STRIP outlines just the nominal
        200x200cm placement zone so it reads as distinct from the search
        margin around it (see occupancy_map.py - that margin isn't a wall,
        just planning headroom)."""
        if self.occ_map is None:
            return

        header_stamp = self.get_clock().now().to_msg()
        height, width = self.occ_map.occupancy_grid.shape
        cell_m = CELL_SIZE_CM / 100.0
        origin = -GRID_MARGIN_CM / 100.0

        line_w = self.get_parameter('grid_line_width_m').value
        line_c = self.get_parameter('grid_line_color').value
        zone_w = self.get_parameter('zone_outline_width_m').value
        zone_c = self.get_parameter('zone_outline_color').value

        lines = Marker()
        lines.header.stamp = header_stamp
        lines.header.frame_id = self.arena_frame
        lines.ns = 'grid_lines'
        lines.id = 0
        lines.type = Marker.LINE_LIST
        lines.action = Marker.ADD
        lines.scale.x = line_w
        lines.pose.orientation.w = 1.0
        lines.color.r, lines.color.g, lines.color.b, lines.color.a = line_c
        # 2026-09-24: only the 200x200cm arena, (0,0) to (2,2) in 10cm cells.
        # This used to span the whole padded planning grid (-1 m to 3 m), which
        # read as a 4x4 m map; the search margin is planning headroom, not arena.
        arena_m = ARENA_SIZE_CM / 100.0
        n_cells = int(round(ARENA_SIZE_CM / CELL_SIZE_CM))
        for i in range(n_cells + 1):
            x = i * cell_m
            lines.points.append(Point(x=x, y=0.0, z=0.01))
            lines.points.append(Point(x=x, y=arena_m, z=0.01))
        for j in range(n_cells + 1):
            y = j * cell_m
            lines.points.append(Point(x=0.0, y=y, z=0.01))
            lines.points.append(Point(x=arena_m, y=y, z=0.01))

        zone = Marker()
        zone.header.stamp = header_stamp
        zone.header.frame_id = self.arena_frame
        zone.ns = 'placement_zone_outline'
        zone.id = 0
        zone.type = Marker.LINE_STRIP
        zone.action = Marker.ADD
        zone.scale.x = zone_w
        zone.pose.orientation.w = 1.0
        zone.color.r, zone.color.g, zone.color.b, zone.color.a = zone_c
        z = 0.015
        zone.points = [
            Point(x=0.0, y=0.0, z=z),
            Point(x=ARENA_SIZE_CM / 100.0, y=0.0, z=z),
            Point(x=ARENA_SIZE_CM / 100.0, y=ARENA_SIZE_CM / 100.0, z=z),
            Point(x=0.0, y=ARENA_SIZE_CM / 100.0, z=z),
            Point(x=0.0, y=0.0, z=z),
        ]

        # Start box: 40x40cm (4x4 cells at the current 10cm/cell resolution),
        # at the arena's own (0,0) corner - same corner the placement zone
        # outline above is drawn from. Same colour/style as that outline
        # (zone_w/zone_c), per direct request - visually groups both as
        # "known fixed zones" rather than needing a new colour convention.
        # NOT the (0.15, 0.15) start POSE used for planning - this is the
        # physical start BOX the car may be placed anywhere within (see
        # control_loop()'s start_pose comment for the pose placeholder's own
        # caveats); the two are independent and this doesn't change planning.
        start_box_size_m = START_BOX_SIZE_CM / 100.0
        start_box = Marker()
        start_box.header.stamp = header_stamp
        start_box.header.frame_id = self.arena_frame
        start_box.ns = 'start_box_outline'
        start_box.id = 0
        start_box.type = Marker.LINE_STRIP
        start_box.action = Marker.ADD
        start_box.scale.x = zone_w
        start_box.pose.orientation.w = 1.0
        start_box.color.r, start_box.color.g, start_box.color.b, start_box.color.a = zone_c
        start_box.points = [
            Point(x=0.0, y=0.0, z=z),
            Point(x=start_box_size_m, y=0.0, z=z),
            Point(x=start_box_size_m, y=start_box_size_m, z=z),
            Point(x=0.0, y=start_box_size_m, z=z),
            Point(x=0.0, y=0.0, z=z),
        ]

        self.grid_marker_pub.publish(MarkerArray(markers=[lines, zone, start_box]))

    def _publish_obstacle_markers(self):
        """Per obstacle: an orange CUBE, a thin red CUBE slab on the facing side,
        the obstacle's number on top of the block, and a small label above it
        with its grid cell (x,y). Published once after planning.

        Drawn directly at the obstacle's own continuous (x_m, y_m) - no
        longer snapped into a grid cell first. Obstacles are now placed as a
        10x10cm (OBSTACLE_SIZE_CM) block CENTRED on the given coordinate
        (see occupancy_map.Obstacle's docstring), so the raw input position
        IS the block's centre - unlike the old grid-floored convention,
        there's no cell-snapping step to reproduce here any more."""
        if not self.obstacles:
            return

        marker_array = MarkerArray()
        header_stamp = self.get_clock().now().to_msg()
        obstacle_c = self.get_parameter('obstacle_color').value
        label_c = self.get_parameter('obstacle_label_color').value
        label_h = self.get_parameter('obstacle_label_height_m').value

        for idx, (x_m, y_m, facing) in enumerate(self.obstacles):
            cx, cy = x_m, y_m

            cube = Marker()
            cube.header.stamp = header_stamp
            cube.header.frame_id = self.arena_frame
            cube.ns = 'obstacles'
            cube.id = idx
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = cx
            cube.pose.position.y = cy
            cube.pose.position.z = 0.05
            cube.pose.orientation.w = 1.0
            cube.scale.x = OBSTACLE_SIZE_CM / 100.0
            cube.scale.y = OBSTACLE_SIZE_CM / 100.0
            cube.scale.z = 0.10
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = obstacle_c
            marker_array.markers.append(cube)

            # Facing: a thin RED slab on the side of the block that carries the
            # image (N/E/S/W as sent by the tablet), proud of the surface by 4 mm.
            half = OBSTACLE_SIZE_CM / 200.0
            thick = 0.008
            fx, fy = {'N': (0.0, 1.0), 'S': (0.0, -1.0),
                      'E': (1.0, 0.0), 'W': (-1.0, 0.0)}[facing]
            face = Marker()
            face.header.stamp = header_stamp
            face.header.frame_id = self.arena_frame
            face.ns = 'obstacle_facing'
            face.id = 300 + idx
            face.type = Marker.CUBE
            face.action = Marker.ADD
            face.pose.position.x = cx + fx * (half + thick / 2.0)
            face.pose.position.y = cy + fy * (half + thick / 2.0)
            face.pose.position.z = 0.05
            face.pose.orientation.w = 1.0
            face.scale.x = thick if fx else OBSTACLE_SIZE_CM / 100.0
            face.scale.y = thick if fy else OBSTACLE_SIZE_CM / 100.0
            face.scale.z = 0.10
            face.color.r, face.color.g, face.color.b, face.color.a = 1.0, 0.0, 0.0, 1.0
            marker_array.markers.append(face)

            # The obstacle's number, drawn on the top of the block itself.
            ident = Marker()
            ident.header.stamp = header_stamp
            ident.header.frame_id = self.arena_frame
            ident.ns = 'obstacle_ids'
            ident.id = 200 + idx
            ident.type = Marker.TEXT_VIEW_FACING
            ident.action = Marker.ADD
            ident.pose.position.x = cx
            ident.pose.position.y = cy
            ident.pose.position.z = 0.102
            ident.scale.z = 0.07
            ident.color.r, ident.color.g, ident.color.b, ident.color.a = 0.0, 0.0, 0.0, 1.0
            ident.text = self._tablet_id(idx)
            marker_array.markers.append(ident)

            # Label above the block: its grid CELL (not cm or metres).
            cell_x = int(math.floor(x_m * 100.0 / CELL_SIZE_CM + 1e-6))
            cell_y = int(math.floor(y_m * 100.0 / CELL_SIZE_CM + 1e-6))
            text = Marker()
            text.header.stamp = header_stamp
            text.header.frame_id = self.arena_frame
            text.ns = 'obstacle_labels'
            text.id = 100 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cx
            text.pose.position.y = cy
            text.pose.position.z = label_h
            text.scale.z = 0.07
            text.color.r, text.color.g, text.color.b, text.color.a = label_c
            text.text = f"({cell_x},{cell_y})"
            marker_array.markers.append(text)

        self.obstacle_marker_pub.publish(marker_array)

    def _publish_checkpoint_markers(self):
        """One ARROW (position + heading) + one TEXT label per obstacle's
        actual planned stand-off checkpoint - the pose obstacle_to_checkpoint()
        computed and each leg's Hybrid A* search (see _plan_all_legs_worker,
        run in a background thread) is asked to reach (see
        collision_aware_planner.plan_visiting_order()'s checkpoints_m). This is
        distinct from the obstacle's own marker: the obstacle marks WHERE
        the target is, this marks WHERE the robot plans to stop and face to
        scan it - previously invisible, only the dense path in between was
        drawn, with no marker for what it was actually aiming at."""
        if not self.checkpoints:
            return

        marker_array = MarkerArray()
        header_stamp = self.get_clock().now().to_msg()
        arrow_c = self.get_parameter('checkpoint_color').value
        arrow_len = self.get_parameter('checkpoint_arrow_length_m').value

        for i, (x, y, theta) in enumerate(self.checkpoints):
            # 1-indexed VISIT-ORDER position (1st stop, 2nd stop, ...), NOT
            # the obstacle's own #N (that's what the /obstacle_markers
            # label already shows) - per direct request: the checkpoint
            # label should show WHEN it gets visited, not WHICH obstacle it
            # belongs to (that's inferrable from proximity/colour to the
            # matching obstacle marker instead).
            visit_pos = i + 1

            arrow = Marker()
            arrow.header.stamp = header_stamp
            arrow.header.frame_id = self.arena_frame
            arrow.ns = 'checkpoints'
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(x)
            arrow.pose.position.y = float(y)
            arrow.pose.position.z = 0.05
            arrow.pose.orientation.z = math.sin(theta / 2.0)
            arrow.pose.orientation.w = math.cos(theta / 2.0)
            arrow.scale.x = arrow_len
            arrow.scale.y = 0.02
            arrow.scale.z = 0.02
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = arrow_c
            marker_array.markers.append(arrow)

            text = Marker()
            text.header.stamp = header_stamp
            text.header.frame_id = self.arena_frame
            text.ns = 'checkpoint_labels'
            text.id = 100 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = 0.18
            text.scale.z = 0.08
            text.color.r, text.color.g, text.color.b, text.color.a = arrow_c
            # No "CP" prefix - checkpoint markers are already colour-coded
            # distinctly from obstacle markers (checkpoint_color vs
            # obstacle_color). Number shown is visit_pos (see above), NOT
            # the obstacle's own #N. .1f, not .2f - see obstacle label
            # above, same 10cm-grid-resolution reasoning.
            text.text = f"#{visit_pos} ({x:.1f}, {y:.1f}, {math.degrees(theta):.0f}deg)"
            marker_array.markers.append(text)

        self.checkpoint_marker_pub.publish(marker_array)

    def _publish_current_path(self):
        """Republishes the full planned-so-far route (see
        _publish_planned_route()) every control tick while navigating - so a
        Foxglove session that connects mid-run still sees everything
        immediately, not just whatever leg happens to be active - PLUS a
        highlight marker for just the currently-active leg specifically
        (own colour/topic), so you can tell which segment the robot is
        actually driving right now within the full route. Does NOT publish
        a single-leg-only Path message any more - that used to overwrite
        _publish_planned_route()'s cumulative one on the SAME topic every
        tick, meaning you'd only ever see one leg at a time regardless of
        how many had actually been planned."""
        self._publish_planned_route()

        if not self.leg_paths or self.current_target_idx >= len(self.leg_paths):
            return
        path = self.leg_paths[self.current_target_idx]
        if not path:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.arena_frame

        # Thick line strip alongside the raw Path - easier to spot in
        # Foxglove's 3D panel (same reasoning as task2_runner.py). Own topic
        # (/path_markers, see occupancy_grid_viz.yaml) - was previously on
        # the same topic as the obstacle markers, confirmed confusing.
        line = Marker()
        line.header = header
        line.ns = 'current_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = self.get_parameter('path_line_width_m').value
        line.color.r, line.color.g, line.color.b, line.color.a = self.get_parameter('path_color').value
        for x, y, *_ in path:   # *_ tolerates (x,y,theta) or (x,y,theta,gear)
            pt = Point()
            pt.x, pt.y, pt.z = float(x), float(y), 0.03
            line.points.append(pt)
        self.path_marker_pub.publish(MarkerArray(markers=[line]))

    def _publish_search_progress(self, points_m):
        """Callback handed to plan_leg()/HybridAStar - called every
        progress_interval node expansions (see hybrid_astar.py's
        find_path()) with every (x, y) point explored SO FAR in the
        currently-running search. Without this, a leg's search is a total
        black box in Foxglove: nothing published for however long it takes,
        no way to tell "still working" from "hung." One POINTS marker,
        replaced in place (fixed id=0) each call rather than accumulating,
        so it always shows the current state of the search, growing as it
        runs - watch it live instead of only seeing the final path once
        (if) the search finishes."""
        msg = Marker()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.arena_frame
        msg.ns = 'search_progress'
        msg.id = 0
        msg.type = Marker.POINTS
        msg.action = Marker.ADD
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.color.r, msg.color.g, msg.color.b, msg.color.a = (1.0, 0.6, 0.0, 0.6)
        msg.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in points_m]
        self.search_progress_pub.publish(MarkerArray(markers=[msg]))

    def _plan_all_legs_worker(self, start_pose_m, gen, leg_paths, checkpoints, occ_map):
        """Runs in a background thread (see the PLANNING_PATH state above,
        which starts it right after plan_visiting_order() and does NOT wait
        for it). Plans every leg back-to-back, one after another, publishing
        each as it finishes - the search for leg 2 can already be running
        while the robot is still driving or paused on leg 1, instead of
        only starting once the robot physically arrives (that was the
        earlier, more conservative "lazy per-leg" design - this is
        eager/continuous instead, per direct request).

        control_loop() (main thread) just polls self.leg_paths[idx] every
        tick and starts driving/skips the moment an entry is filled in -
        see NAVIGATING_TO_TARGET's handling. The only shared mutable state
        this thread touches is single-index assignments into the plain list
        self.leg_paths (atomic under the GIL - one writer here, one reader
        in control_loop(), no lock needed) and rclpy publisher calls
        (.publish() is documented safe to call from any thread). Wrapped in
        try/except so a genuine crash mid-search logs an error instead of
        silently hanging the whole route forever with no explanation."""
        try:
            leg_start_pose_m = start_pose_m
            for idx, target_pose_m in enumerate(checkpoints):
                if gen != self._plan_gen:
                    return   # abandoned (new setup / stop) - write nothing
                path = plan_leg(
                    occ_map, leg_start_pose_m, target_pose_m,
                    theta_offset=TASK1_CAMERA_THETA_OFFSET_RAD,
                    progress_callback=self._publish_search_progress)
                if gen != self._plan_gen:
                    return
                leg_paths[idx] = path
                status = f"{len(path)} points" if path else "NO PATH FOUND"
                self.get_logger().info(f"Leg {idx + 1}/{len(checkpoints)} planned: {status}")
                self._publish_planned_route()
                leg_start_pose_m = target_pose_m
            if gen == self._plan_gen:
                self.plan_state = 'DONE'
                self.get_logger().info("All legs planned.")
        except Exception:
            # rclpy's logger takes a plain string, not stdlib logging's
            # exc_info kwarg - format the traceback in manually.
            self.get_logger().error(f"Background leg-planning thread crashed:\n{traceback.format_exc()}")

    def _start_current_leg(self) -> bool:
        """Loads the current target's dense path into the follower, IF the
        background planning thread (_plan_all_legs_worker) has finished
        computing it yet. Returns True once there's something definite to
        say about this leg (a real path was loaded, or it was confirmed
        empty and the state machine already moved on) - False only while
        still waiting on the background thread, so the caller knows to keep
        polling rather than getting stuck.

        An empty leg (Hybrid A* found no path - see collision_aware_planner
        docstring) transitions straight to PAUSE_FOR_SCAN itself, right
        here - not left for the caller to notice, since self.follower.active
        would never become True for an empty path (bool([]) is False), and
        looping on that would wait forever instead of skipping."""
        idx = self.current_target_idx
        path = self.leg_paths[idx]
        if path is None:
            return False   # still being computed - see _plan_all_legs_worker

        self.follower.set_path(path)
        if not path:
            self.get_logger().warn(f"No path found to target {idx + 1}, skipping navigation.")
            self.detected_target_id = None
            self.state = State.PAUSE_FOR_SCAN
            self.state_start_time = self.get_now_sec()
        return True

    def _publish_planned_route(self):
        """Publishes the FULL route planned SO FAR - every leg
        _plan_all_legs_worker has finished, concatenated in visit order -
        as one /planned_path message, called again each time a new leg
        completes. NOT one Path message per leg: a nav_msgs/Path panel in
        Foxglove shows only the latest message it received, it doesn't
        accumulate multiple Path messages into one view - publishing each
        leg as its own separate message would mean you only ever see
        whichever leg finished MOST RECENTLY, never the whole route
        building up (confirmed - this was the bug behind only ever seeing
        leg 1's path). Republishing the whole thing every time a leg
        finishes is cheap (at most a few hundred points total) and means
        the displayed path always reflects everything planned up to now."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.arena_frame
        for path in self.leg_paths:
            if not path:
                continue
            for x, y, *_ in path:   # *_ tolerates (x,y,theta) or (x,y,theta,gear)
                p = PoseStamped()
                p.header = msg.header
                p.pose.position.x = float(x)
                p.pose.position.y = float(y)
                p.pose.position.z = 0.05
                msg.poses.append(p)
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Task1Runner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
