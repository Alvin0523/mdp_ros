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
from geometry_msgs.msg import TwistStamped, PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

from mdp_algorithm.collision_aware_planner import plan_leg, plan_visiting_order
from mdp_algorithm.occupancy_map import (
    ARENA_SIZE_CM, CELL_SIZE_CM, GRID_MARGIN_CM, OBSTACLE_SIZE_CM,
    Obstacle, OccupancyMap,
)
from mdp_algorithm.pure_pursuit_follower import PurePursuitController, yaw_from_quaternion

# Task 1's physical camera is mounted facing the car's LEFT side, not
# forward - confirmed by the user, not modeled in mini_akm_robot.urdf's
# camera_joint (rpy="0 0 0", forward-facing - that matches Task 2's own
# camera config instead, see task2_runner.py). This is what
# hamiltonian.obstacle_to_checkpoint()'s theta_offset corrects for: the
# checkpoint's body heading is chosen so the CAMERA (not the front
# bumper) ends up pointed at the obstacle's face.
#
# SIGN FLIPPED 2026-09-04: +pi/2 (the naive "positive=left/REP-103" guess)
# produced checkpoints that visually pointed the RIGHT side at the
# obstacle in Foxglove, confirmed by direct observation - the
# obstacle_to_checkpoint() formula's theta = image_bearing - theta_offset
# resolves the opposite way round from that naive guess. -pi/2 is the
# corrected value; re-verify against Foxglove (checkpoint arrow direction
# vs. which physical side the camera is on) before trusting this for a
# real run - this was flipped once already on inherited-formula guesswork,
# not re-derived from scratch.
TASK1_CAMERA_THETA_OFFSET_RAD = -math.pi / 2.0

# Physical start box the car may be placed anywhere within (per direct user
# description) - 40x40cm, at the arena's own (0,0) corner. Visualization
# only (see _publish_grid_lines' start_box marker) - does NOT feed the
# (0.15, 0.15) start POSE used for planning below, which is a separate,
# still-unverified placeholder for exactly where inside this box the car is
# assumed to start.
START_BOX_SIZE_CM = 40.0


class State(Enum):
    WAITING_FOR_SETUP = auto()
    PLANNING_PATH = auto()
    NAVIGATING_TO_TARGET = auto()
    PAUSE_FOR_SCAN = auto()
    FINISHED = auto()


class Task1Runner(Node):
    def __init__(self):
        super().__init__('task1_runner')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        # Visualization style/topic-name parameters - see
        # mdp_bringup/config/occupancy_grid_viz.yaml for the full set and
        # its scope note (visualization only, not the underlying grid math).
        self._declare_viz_params()

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

        # Control Loop @ 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)

        # Planner & path-following state
        self.follower = PurePursuitController()
        self.current_pose = (0.0, 0.0, math.pi / 2)  # x, y, yaw - updated by odom_callback
        self.state = State.WAITING_FOR_SETUP

        self.obstacles = []          # (x_m, y_m, facing) as received from /obstacle_setup
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

        self.detected_target_id = None
        self.state_start_time = self.get_now_sec()
        self.get_logger().info("Task 1 Runner Node Initialized! Waiting for obstacle setup...")

    def _declare_viz_params(self):
        """Declares every parameter in occupancy_grid_viz.yaml with the same
        defaults that file documents, so this node runs sensibly even if
        launched without that config (e.g. ros2 run directly, as done for
        local testing) - the YAML overrides these when loaded via launch."""
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

    def get_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = (p.x, p.y, yaw)
        self.follower.update_pose(p.x, p.y, yaw)

    def setup_callback(self, msg: String):
        if self.state == State.WAITING_FOR_SETUP:
            self.obstacles.clear()
            raw_items = msg.data.strip().split('|')
            for item in raw_items:
                if ':' in item:
                    obs_id, data = item.split(':')
                    parts = data.split(',')
                    x, y = float(parts[0]), float(parts[1])
                    face = parts[2]
                    self.obstacles.append((x, y, face))

            self.get_logger().info(f"Loaded {len(self.obstacles)} obstacles from setup!")
            self.state = State.PLANNING_PATH

    def yolo_callback(self, msg: String):
        if self.state == State.PAUSE_FOR_SCAN and self.detected_target_id is None:
            self.detected_target_id = msg.data.strip()
            self.get_logger().info(f"YOLO26 Identified Target: {self.detected_target_id}")

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
            start_pose = (0.15, 0.15, math.pi / 2)

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
                    f"Obstacles with no valid scan checkpoint, skipped: {[i + 1 for i in self.unreachable]}")
            self.get_logger().info(f"Visiting Order Calculated: {[i + 1 for i in self.visiting_order]}")

            self.leg_paths = [None] * len(self.visiting_order)
            self.current_target_idx = 0
            self.state = State.NAVIGATING_TO_TARGET
            self.state_start_time = now

            # Plans every leg back-to-back, in the background, starting NOW
            # - not gated on the robot physically reaching each checkpoint
            # first (see this thread's own docstring). NAVIGATING_TO_TARGET
            # below picks up leg 0 the moment it's ready, same as before;
            # every later leg just gets a head start instead of only
            # starting once the robot arrives at the checkpoint before it.
            self._planning_thread = threading.Thread(
                target=self._plan_all_legs_worker, args=(start_pose,), daemon=True)
            self._planning_thread.start()

        # STATE 2: Navigating to current target standoff pose - now actually
        # tracks the Hybrid A*-planned path via PurePursuitController,
        # instead of blindly driving forward for a fixed 2.5s.
        elif self.state == State.NAVIGATING_TO_TARGET:
            # Leg may still be mid-search in the background thread (see
            # _plan_all_legs_worker) - check every tick, not just once at
            # the state transition, so the robot starts driving the instant
            # it's ready instead of only when arrival at the PREVIOUS
            # checkpoint happened to trigger a (re)check.
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
                self.send_bt(f"ROBOT,{self.current_pose[0]:.2f},{self.current_pose[1]:.2f},"
                             f"{math.degrees(self.current_pose[2]):.0f}")

        # STATE 3: Pause for YOLO26 scanning & Bluetooth update
        elif self.state == State.PAUSE_FOR_SCAN:
            self.send_cmd(0.0, 0.0)

            if self.detected_target_id is not None or elapsed > 0.6:
                target_id = self.detected_target_id if self.detected_target_id else "UNKNOWN"
                obs_num = self.visiting_order[self.current_target_idx] + 1

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
        msg.header.frame_id = 'odom'
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
        """Explicit cell-boundary lines across the full padded grid, not
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
        lines.header.frame_id = 'odom'
        lines.ns = 'grid_lines'
        lines.id = 0
        lines.type = Marker.LINE_LIST
        lines.action = Marker.ADD
        lines.scale.x = line_w
        lines.pose.orientation.w = 1.0
        lines.color.r, lines.color.g, lines.color.b, lines.color.a = line_c
        for i in range(width + 1):
            x = origin + i * cell_m
            lines.points.append(Point(x=x, y=origin, z=0.01))
            lines.points.append(Point(x=x, y=origin + height * cell_m, z=0.01))
        for j in range(height + 1):
            y = origin + j * cell_m
            lines.points.append(Point(x=origin, y=y, z=0.01))
            lines.points.append(Point(x=origin + width * cell_m, y=y, z=0.01))

        zone = Marker()
        zone.header.stamp = header_stamp
        zone.header.frame_id = 'odom'
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
        start_box.header.frame_id = 'odom'
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
        """One CUBE + one TEXT_VIEW_FACING label per obstacle, published
        once after planning (positions are static for the run). Label shows
        the 1-based obstacle number and its facing side, since that facing
        is what determines the checkpoint pose and is otherwise invisible
        in the visualization.

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
            cube.header.frame_id = 'odom'
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

            text = Marker()
            text.header.stamp = header_stamp
            text.header.frame_id = 'odom'
            text.ns = 'obstacle_labels'
            text.id = 100 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cx
            text.pose.position.y = cy
            text.pose.position.z = label_h
            text.scale.z = 0.10
            text.color.r, text.color.g, text.color.b, text.color.a = label_c
            # .1f, not .2f - matches the arena's actual 10cm grid resolution
            # (CELL_SIZE_CM), a 2nd decimal place is never meaningful here.
            text.text = f"#{idx + 1} [{facing}] ({x_m:.1f}, {y_m:.1f})"
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
            arrow.header.frame_id = 'odom'
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
            text.header.frame_id = 'odom'
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
        header.frame_id = 'odom'

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
        for x, y, _ in path:
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
        msg.header.frame_id = 'odom'
        msg.ns = 'search_progress'
        msg.id = 0
        msg.type = Marker.POINTS
        msg.action = Marker.ADD
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.color.r, msg.color.g, msg.color.b, msg.color.a = (1.0, 0.6, 0.0, 0.6)
        msg.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in points_m]
        self.search_progress_pub.publish(MarkerArray(markers=[msg]))

    def _plan_all_legs_worker(self, start_pose_m):
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
            for idx, target_pose_m in enumerate(self.checkpoints):
                path = plan_leg(
                    self.occ_map, leg_start_pose_m, target_pose_m,
                    theta_offset=TASK1_CAMERA_THETA_OFFSET_RAD,
                    progress_callback=self._publish_search_progress)
                self.leg_paths[idx] = path
                status = f"{len(path)} points" if path else "NO PATH FOUND"
                self.get_logger().info(f"Leg {idx + 1}/{len(self.checkpoints)} planned: {status}")
                self._publish_planned_route()
                leg_start_pose_m = target_pose_m
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
        msg.header.frame_id = 'odom'
        for path in self.leg_paths:
            if not path:
                continue
            for x, y, _ in path:
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
