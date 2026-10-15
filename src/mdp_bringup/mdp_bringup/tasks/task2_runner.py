"""
Task 2 runner: out of the carpark, past obstacle 1 on its arrow's side, round
obstacle 2 on its arrow's side, back into the carpark - as fast as it can.

The distances (obstacle 1 at 0.6-1.5 m, obstacle 2 another 0.6-1.5 m) are not
known in advance, so the car measures them with the front ultrasonic:

  Four checkpoints: 1 beside obstacle 1 (arrow 1's side), 2 beside obstacle 2's
  end on arrow 2's side, 3 beside its other end (round the back), 4 = home.

  WAITING_FOR_START   standing in the carpark: the ultrasonic measures obstacle 1,
                   YOLO reads arrow 1 -> checkpoint 1 (heading as at home) and the
                   path to it are planned before START
  TO_CHECKPOINT_1  START: that path - straight, a smooth S (its curvature rises
                   and falls gradually, task2_paths.py), straight into checkpoint 1,
                   arriving pointing straight. On the way YOLO reads arrow 2 (the
                   arrow well beyond obstacle 1's face, task2_vision.py); on the
                   straight bit (cp1_speed, from line_up_dist before checkpoint 1)
                   the ultrasonic measures obstacle 2 (a bar on the centre line)
                   and the camera sees arrow 2 and a bullseye on its near end
  CHECKPOINT_1     the bar's width from the gap between arrow 2 and that bullseye
                   in one frame x the ultrasonic's distance (else bullseye +
                   ultrasonic + pose; the usual 60 cm if none was seen), to the
                   nearest 5 cm. Something missing: stop at checkpoint 1 and read
                   up to CP1_WAIT_S - no rolling on or turning ->
                   checkpoints 2, 3, 4; leg 1 (CP1 -> CP2) from the route table
                   built offline (task2_routes.py), the others clean geometry
                   (milliseconds). Hybrid A* - on the laptop with one - only for
                   what none of that covers. Stops only if a reading is missing
  FOLLOW           1 -> 2 an S, straight into 2; 2 -> 3 half a circle round the
                   back of the bar; 3 -> 4 an S onto the centre line, straight in.
                   Forward only: the follower never reverses on its own, only a
                   planned reverse or the car stuck for stuck_time s
  (fallback, arrow 1 not read at home: straight on until YOLO reads it, then the
   same S to checkpoint 1 from there - APPROACH_1; too close for that:
   SWERVE_OUT/BACK, full lock out and back)
  FINISHED         in the carpark (or the ultrasonic sees the back wall)

step:=true (launch argument, testing): one checkpoint per /step (`pixi run step`,
SPACE). The first starts the run; the car stops at checkpoint 1, stands still
STEP_LOOK_S reading arrow 2, the bar's distance and width, then plans - or says
what is missing and stays (SPACE looks again). Each further SPACE drives one leg
and stops at its checkpoint. No path: it stays. Nothing is guessed.
A live preview at home and at checkpoint 1, for trying layouts: every LIVE_REFRESH_S
it reads again from scratch and re-plans what changed, drawn in Foxglove - at home
obstacle 1 and arrow 1 (the path to checkpoint 1), at checkpoint 1 arrow 2 and the
bar's distance and width (checkpoints 2, 3, 4 and their legs). Move or swap an
obstacle and watch. SPACE commits the latest plan and drives; nothing is re-planned
after that.

Frame (`map`): the task 2 course, x along it, centre line y = 1.2 m - see
utils/obstacle_layout.py. The course rules (carpark, obstacle sizes) come from
config/tasks.yaml task2. Settings: config/navigation.yaml task2_runner. Drawn
in Foxglove like task 1.

`pixi run setup` (Foxglove SETUP): tasks.yaml's task2 `setup` layout arrives on
/task2_setup - both obstacles and arrows KNOWN, so no ultrasonic measuring and
no YOLO: checkpoint 1 is planned before START and checkpoints 2-4 at checkpoint 1
from the known bar. Kept across resets, like task 1's set. Served by the
laptop's task1_planner (role:=pi) or by this node (serve_setup).
"""

import dataclasses
import json
import math
import os
import statistics
import threading
import time
import traceback
from collections import Counter, deque
from enum import Enum, auto

import rclpy.qos
import rclpy.time
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from mdp_interfaces.msg import RunStatus
from sensor_msgs.msg import Range
from std_msgs.msg import String
from std_srvs.srv import Trigger

from mdp_algorithm.control.path_follower import PathFollower, yaw_from_quaternion
from mdp_algorithm.planning import task2_paths
from mdp_algorithm.planning.planner import plan_leg
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils import config, markers, obstacle_layout
from mdp_bringup.utils.obstacle_layout import TASK2_AREA_M, TASK2_BACK_WALL_X, TASK2_CENTRE_Y
from mdp_bringup.tasks import task2_routes, task2_vision
from mdp_bringup.tasks.ir_face_pass import FacePass
from mdp_bringup.tasks.runner_base import RunnerBase
from mdp_bringup.utils.run import run

ARROWS = {'39': 'LEFT', 'LEFT': 'LEFT', '38': 'RIGHT', 'RIGHT': 'RIGHT'}   # YOLO id -> side
SIDE = task2_routes.SIDE                                                  # +y = left
STUCK_SPEED = 0.03            # m/s: slower than this while told to drive = not moving
POSE_HISTORY_S = 2.0          # s of poses kept, to place a camera frame by its time stamp
LIVE_PLAN_LEAD_M = 0.10       # m further on for a leg 1 planned live (the search takes ~0.1-0.5 s)
ROLL_ON_M = 0.50              # m before the bar's face the car may still roll on, leg 1 not planned yet
BAR_X_AGREE_M = 0.12         # m the ultrasonic's and the camera's distance to the bar may differ
CP1_WINDOW_M = (-0.12, 0.40)  # m along: the car this near checkpoint 1 starts leg 1 from the route table.
                              # It "arrives" up to ~8 cm short: the follower allows for the pose's
                              # latency (pose_latency 0.09 s x 0.9 m/s, sim 2026-10-14)
HEADING_GAIN = 1.5            # rad of steering per rad of heading error, on straights
ARC_LEAD_RAD = math.radians(5.0)   # end an arc this early: the steering takes a moment to swing back
ARROW_WAIT_S = 3.0            # an arrow not read yet when needed: wait this long, then guess
STEP_LOOK_S = 2.0             # step:=true: s standing at checkpoint 1 collecting boxes and ultrasonic
LIVE_REFRESH_S = 5.0          # step:=true: s between fresh reads at home and at checkpoint 1 (live preview)
US_VALID = (0.25, 2.2)        # m - lane readings outside this are not obstacle 2
# Checkpoint 1 with arrow 2, obstacle 2's distance or the bar's width still missing:
# stop there and read this long. Rolling on angled toward the centre (7 deg) or a 15
# deg glance wiggled the car +-15 deg and 20-56 cm past checkpoint 1, off the route
# table's start: leg 1 planned live, twice NO PATH (bags 2026-10-15).
CP1_WAIT_S = 1.5
# ...but first roll on straight this far past it at cp1_speed: YOLO's boxes come ~0.2 s
# after the frame, and arrow 2 usually arrives within that (it stopped for 0.2-0.4 s in
# most runs, 2026-10-16). Straight, so leg 1's table route still starts from here.
CP1_ROLL_M = 0.0              # no roll: stop at checkpoint 1 (rolling on toward a near bar was too risky,
                              # 2026-10-16; 0.15 / 0.35 tried)
WEAK_WIDTH_DIST_M = 1.0       # m from the ultrasonic to the bar: further, its width is planned at least usual
LEVEL_X_M = 0.30              # m before obstacle 1's centre (or the straight into checkpoint 1, if
                              # earlier: level_with_1()): straight in the lane from here, the
                              # ultrasonic and camera see past it (obstacle 2 and arrow 2) and
                              # arrow 1 is out of the camera's view - read them on the move
BRAKE_RESPONSE_S = 0.15       # s for the wheels' speed to follow a slower command (speed_ahead)
IN_CARPARK_MARGIN_M = 0.06    # m: the whole outline this far inside the carpark's walls = home (0.02 until
                              # 2026-10-16: it stopped just past the entrance line)
NO_BAR_LIMIT_M = 0.35         # m past obstacle 1's centre without obstacle 2 seen: stop (the
                              # bar's face can be 0.65 m past it, the car's nose is 0.2 m ahead)


class State(Enum):
    WAITING_FOR_START = auto()
    TO_CHECKPOINT_1 = auto()
    APPROACH_1 = auto()
    SWERVE_OUT = auto()
    SWERVE_BACK = auto()
    CHECKPOINT_1 = auto()
    FOLLOW = auto()
    FINISHED = auto()
    STOPPED = auto()


class Task2Runner(RunnerBase):
    def __init__(self):
        super().__init__('task2_runner')
        # straight_speed, path_speed, swerve_trigger_dist, lane_offset, side_clearance,
        # home_stop_dist, ...: config/navigation.yaml (live).
        config.declare(self, 'task2_runner')
        # Task 2's own margin round the obstacles (obstacle_padding) in place of the shared
        # costmap.footprint_padding - this process only, so task 1 keeps its own.
        planner_params.configure(dataclasses.replace(
            planner_params.ACTIVE, footprint_padding=float(self.p('obstacle_padding'))))
        layout = self.declare_parameter(
            'layout', f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml").value
        self.arena = obstacle_layout.load_task2(layout)
        self.layout = layout
        # (x1, x2, arrow 1, arrow 2) from `pixi run setup`, None = measure them. Not
        # cleared by reset_run(): a reset readies the same layout for the next run.
        self.known_layout = None
        self.reset_done = False          # RESET:DONE to the tablet: a reset since the last run
        self._last_sent, self._last_heartbeat = {}, 0.0
        self.create_subscription(String, '/task2_setup', self.task2_setup_callback, 10)
        self.task2_setup_pub = self.create_publisher(String, '/task2_setup', 10)
        # With a laptop (role:=pi) its task1_planner serves /setup_obstacles from the
        # laptop's tasks.yaml, for both tasks; alone, this node does.
        if self.declare_parameter('serve_setup', True).value:
            self.create_service(Trigger, '/setup_obstacles', self.setup_obstacles_callback)
        # step:=true - one checkpoint per /step (module docstring).
        self.step = bool(self.declare_parameter('step', False).value)
        self.create_service(Trigger, '/step', self.step_callback)
        # What YOLO saw at each obstacle - the laptop's scan_snapshots saves those frames.
        self.snap_pub = self.create_publisher(String, '/task2_snapshot', 10)
        self._ready_note = None          # the last READY / plan-failure line (each logged once)
        self.car =planner_params.ACTIVE          # URDF + navigation.yaml (robot.*, follower.*)
        car_centre_ahead = (self.car.footprint_front - self.car.footprint_rear) / 2.0
        self.home = self.arena.car_pose_centred(car_centre_ahead, math.pi)
        # Where every run starts (the launch's start pose: the carpark, facing out) - the
        # path to checkpoint 1 is planned from here, like task 1, not from the live pose.
        default = self.arena.car_pose_centred(car_centre_ahead, 0.0)
        self.start = tuple(float(self.declare_parameter(n, v).value)
                           for n, v in zip(('start_x', 'start_y', 'start_yaw'), default))

        # (/cmd_vel, /run_status, the Foxglove drawings, /start_run ...: RunnerBase)
        self.create_subscription(Range, '/ultrasonic', self.ultrasonic_callback, 10)
        # The IRs at the front wheels, one each side (URDF task 2): position fixes
        # while the car passes obstacle 1's side and the bar's ends (ir_face_pass.py).
        for name in ('ir', 'ir2'):
            self.create_subscription(Range, f'/{name}', lambda m, n=name: self.ir_callback(n, m), 10)
        self.ir_sensors = {}             # name -> (x, y, yaw) on base_link (TF, from the URDF)

        self.follower = PathFollower()
        # Task 2 sets the speed and lookahead itself (speed_ahead(), follower_step()).
        self.follower.regulate = self.follower.velocity_scaled_lookahead = False
        self.us_offset = None            # base_link -> ultrasonic_link, m ahead (TF, from the URDF)
        self.cam_offset = None           # base_link -> camera_link (x, y, yaw) (TF, from the URDF)

        # Every box YOLO sees (arrows, bullseyes), and the laptop's bullseye_matcher's
        # bullseyes in the same format until YOLO knows them (task2_vision.py).
        self.create_subscription(String, '/yolo_detections', lambda m: self.detections_callback(m, 'yolo'), 10)
        self.create_subscription(String, '/bullseye_detections', self.detections_callback, 10)
        self.boxes_seen_at = None        # when any box list last arrived (YOLO / matcher running)
        self.poses = deque()             # (time s, pose) - where the car was when a frame was taken

        # Checkpoint 1 -> 2 from the route table (task2_routes.py), if it was built with
        # these settings; else planned live like the other legs.
        self.routes = self.load_routes()

        # Hybrid A* on the laptop (task1_planner) with one (role:=pi), else here.
        self.remote_planner = self.declare_parameter('remote_planner', False).value
        plan_qos = rclpy.qos.QoSProfile(depth=20, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)
        self.plan_request_pub = self.create_publisher(String, '/plan_legs/request', plan_qos)
        self.create_subscription(String, '/plan_legs/result', self.plan_result_callback, plan_qos)
        self._leg_waits = {}             # gen -> {'ack', 'done', 'path'} (remote_leg)
        self._remote_n = 0
        self.reset_run()
        self.publish_map()
        self.get_logger().info("READY     waiting for START")

    def reset_run(self):
        self.set_state(State.WAITING_FOR_START)
        self.start_received = False
        self.run_start = self.run_end = None
        self.current_pose = (0.0, 0.0, 0.0)
        self.pose_offset = (0.0, 0.0)    # the IR fixes of the last run
        self.have_pose = False           # START waits for the first EKF pose
        self.us_range = None             # (range m, time s) - latest valid ultrasonic reading
        self.us_trigger_count = 0
        self.arrow_counts = [Counter(), Counter()]   # per obstacle
        self.arrows = [None, None]
        self._arrow_wait_since = None
        self._cp1_wait_since = None      # stopped at checkpoint 1 for a missing reading since
        self._cp1_note = None            # what it waits for (logged once)
        self.step_go = False             # step:=true: a /step came - the next leg (or another look) may go
        self._look_since = None          # step:=true: standing at checkpoint 1 since
        self._step_note = None           # step:=true: the last "waiting" line (logged once)
        self._looks = 0                  # step:=true: looks done at checkpoint 1 (the first is STEP_LOOK_S)
        self._preview = None             # step:=true: (x2, why, arrow 2, bar width) planned at checkpoint 1
        self._home_look_since = None     # step:=true: arrow 1's reads at home counted since
        self._bar_info = (None, 'setup')  # (measured width m or None, how) - for the OBSTACLE 2 line
        self._preview_src = ''           # step:=true: where the previewed bar distance came from
        self._leg1_note = ''             # leg 1: from the route table, or live and why
        self._arrow1_stamp = 0.0         # the frame arrow 1 was last read in (snapshot)
        self.side = 0.0                 # +1 left / -1 right, the lane beside obstacle 1
        self.x1 = self.x2 = None         # obstacle centres (m, along the course)
        self.lane_readings = []
        self.waypoints = []              # (x, y, theta): A, B, HOME
        self.names = ['checkpoint 2', 'checkpoint 3', 'checkpoint 4 (home)']
        self.leg_paths = []
        self.leg = 0
        self.costmap = None
        # The path start -> beside obstacle 1, planned before START from self.start. With the
        # setup layout it stays planned across resets (a reset only puts the pose back).
        pre = getattr(self, 'pre', None)
        self.pre = pre if self.known_layout and pre and pre.get('path') else None
        self.lane_waypoint = self.pre['goal'] if self.pre else None   # checkpoint 1, beside obstacle 1
        self.x1_seen = None              # obstacle 1 as the ultrasonic sees it from home (drawn)
        self.passes = []                 # (sensor name, FacePass) - faces the IRs are watching for
        self._plan_gen = getattr(self, '_plan_gen', 0) + 1
        self.bullseyes = []              # (frame time, box, image w, camera pose) seen near checkpoint 1
        self.arrows_2 = []               # (frame time, box) arrow 2's boxes there (the bar's width, gap method)
        self.arrow_scales = getattr(self, 'arrow_scales', [])   # YOLO arrow box / card, measured at home
        self.bar_width = None            # m, the bar's width the run plans with (nearest 5 cm), once known
        self.cp1_planned = False         # lane_waypoint is the planned checkpoint 1 (not a swerve's end)
        self._stuck_since = None
        self.set_path([])

    # ------------------------------------------------------------ helpers ----

    def p(self, name):
        return self.get_parameter(name).value

    def steer(self, v: float, steering: float):
        """Drive at v with the front wheels at `steering` rad (+ left), clamped to
        the car's limits - bicycle model, as ackermann_steering_controller."""
        steering = max(-self.car.steering_limit_right, min(self.car.steering_limit_left, steering))
        self.send_cmd(v, v * math.tan(steering) / self.car.wheelbase)

    def hold_heading(self, v: float, heading: float):
        err = math.atan2(math.sin(self.current_pose[2] - heading), math.cos(self.current_pose[2] - heading))
        self.steer(v, -HEADING_GAIN * err)

    def set_path(self, path):
        """Hand the follower a path. A path without a reverse is driven forward only:
        the follower does not reverse on its own (it backed the car up after
        checkpoint 1, car 2026-10-14) - only check_stuck() lets it."""
        self.follower.set_path(path)
        self.follower.forward_only = bool(path) and not task2_paths.has_reverse(path)
        self.follower.behind_speed = float(self.p('path_speed'))
        self._stuck_since = None

    def check_stuck(self):
        """Told to drive but not moving for stuck_time s: let the follower reverse
        (a forward-only path) - the one case the car may back up on its own."""
        if not self.follower.forward_only or self.last_odom is None:
            self._stuck_since = None
            return
        moving = abs(self.last_odom.twist.twist.linear.x) >= STUCK_SPEED
        if moving or abs(self._last_cmd[0]) < 0.1:
            self._stuck_since = None
            return
        if self._stuck_since is None:
            self._stuck_since = self.now()
        elif self.now() - self._stuck_since >= float(self.p('stuck_time')):
            self.follower.forward_only = False
            self.get_logger().warn(f"STUCK     at {self.fmt(self.current_pose)} for "
                                   f"{float(self.p('stuck_time')):.1f} s - reversing allowed on this leg")

    # ------------------------------------------------------- route table ----

    def route_fingerprint(self):
        settings = {name: self.p(name) for name in task2_routes.ROUTE_SETTINGS}
        return task2_routes.fingerprint(self.car, self.arena, settings)

    def load_routes(self):
        """config/task2_routes.json.gz, if built with these settings (else None: live)."""
        path = self.declare_parameter(
            'route_table', os.path.join(get_package_share_directory('mdp_bringup'), 'config',
                                        task2_routes.TABLE_FILE)).value
        try:
            table = task2_routes.RouteTable.load(path)
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().warn(f"ROUTES    {path} unreadable ({exc}) - checkpoint 1 -> 2 planned live")
            return None
        if table is None:
            self.get_logger().warn(f"ROUTES    no {path} - checkpoint 1 -> 2 planned live (pixi run task2-table)")
            return None
        if not table.matches(self.route_fingerprint()):
            self.get_logger().warn("ROUTES    the route table was built with other settings - checkpoint 1 -> 2 "
                                   "planned live (rebuild: pixi run task2-table)")
            return None
        return table

    def routes_ok(self) -> bool:
        """The table still matches the settings (a live `ros2 param set` may change them)."""
        return self.routes is not None and self.routes.matches(self.route_fingerprint())

    # ------------------------------------------------------- camera boxes ----

    def camera_offset(self):
        """base_link -> camera_link (x, y, yaw), from the URDF via TF."""
        if self.cam_offset is None:
            try:
                t = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time()).transform
                self.cam_offset = (t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation))
            except tf2_ros.TransformException:
                return (0.171, 0.0, 0.0)     # URDF task 2: 28 mm behind the chassis front, centre line
        return self.cam_offset

    def camera_pose(self, pose):
        cx, cy, cyaw = self.camera_offset()
        x, y, yaw = pose
        return (x + cx * math.cos(yaw) - cy * math.sin(yaw), y + cx * math.sin(yaw) + cy * math.cos(yaw), yaw + cyaw)

    def on_pose(self):
        now = self.now()
        self.poses.append((now, self.current_pose))
        while self.poses and now - self.poses[0][0] > POSE_HISTORY_S:
            self.poses.popleft()

    def pose_at(self, t: float):
        """The car's pose at time t (the nearest kept), or None if not kept."""
        if not self.poses or t < self.poses[0][0] - 0.1:
            return None
        return min(self.poses, key=lambda tp: abs(tp[0] - t))[1]

    def detections_callback(self, msg: String, source: str = 'matcher'):
        """/yolo_detections, /bullseye_detections: every box of a frame. Arrows ->
        arrow 1 or 2 by their distance (task2_vision.classify_arrow); bullseyes near
        checkpoint 1 -> kept for the bar's width (measure_bar())."""
        try:
            stamp, w, _, boxes = task2_vision.parse(msg.data)
        except (ValueError, KeyError, TypeError):
            return
        self.boxes_seen_at = self.now()
        if self.state not in (State.WAITING_FOR_START, State.APPROACH_1, State.TO_CHECKPOINT_1, State.CHECKPOINT_1):
            return
        pose = self.pose_at(stamp) if self.state != State.WAITING_FOR_START else self.current_pose
        if pose is None:
            return
        hfov = float(self.p('camera_hfov'))
        cam = self.camera_pose(pose)
        arrows = [b for b in boxes if b[0] in task2_vision.ARROW_IDS]
        if arrows:
            if self.state in (State.WAITING_FOR_START, State.APPROACH_1):
                # Before turning out: the biggest (nearest) arrow is obstacle 1's - and,
                # obstacle 1's distance known (ultrasonic), how big YOLO boxes an arrow.
                near = max(arrows, key=lambda b: b[5] - b[3])
                self.count_arrow(0, task2_vision.ARROW_IDS[near[0]])
                self._arrow1_stamp = stamp
                x1 = self.home_x1() if self.state == State.WAITING_FOR_START else None
                if x1 is not None:
                    d = x1 - self.arena.obstacle_1_size[0] / 2.0 - cam[0]
                    if d > 0.2:
                        self.arrow_scales.append(task2_vision.arrow_scale(near, w, hfov, d))
                        del self.arrow_scales[:-20]
            else:
                face1 = self.x1 - self.arena.obstacle_1_size[0] / 2.0 if self.x1 is not None else None
                scale = task2_vision.median(self.arrow_scales) or task2_vision.ARROW_BOX_SCALE
                for b in arrows:
                    i = task2_vision.classify_arrow(b, w, hfov, cam[0], face1, scale)
                    if i == 1 or self.level_with_1():
                        self.count_arrow(1, task2_vision.ARROW_IDS[b[0]])
                        if self.near_checkpoint_1(pose):
                            self.arrows_2.append((stamp, b))
        # Bullseyes from the end of the S into checkpoint 1 on: the bar is near and big in
        # the picture there (sim 2026-10-14: 0.2-0.3 cm off from checkpoint 1; from the
        # carpark up to 7 cm). The camera's pose comes with the frame, so a few degrees of
        # turn left are fine; only within 10 deg of straight left one frame a run.
        if self.near_checkpoint_1(pose):
            for b in boxes:
                if b[0] == task2_vision.BULLSEYE_ID:
                    self.bullseyes.append((stamp, b + [source], w, cam))   # merged per frame (bar_width)

    def near_checkpoint_1(self, pose) -> bool:
        return (self.state in (State.TO_CHECKPOINT_1, State.CHECKPOINT_1) and self.x1 is not None
                and pose[0] >= self.x1 - 2 * LEVEL_X_M and abs(pose[2]) < math.radians(25.0))

    def count_arrow(self, i: int, side: str):
        if self.arrows[i] is None:
            self.arrow_counts[i][side] += 1

    def measure_bar(self, x2: float):
        """The bar's width from the frames seen near checkpoint 1: (width or None,
        widths, 'gap') - arrow 2 to bullseye in one frame only. The pose method (one
        bullseye + the car's own heading and position) read 42-45 cm for real bars (car
        2026-10-16): not used - the car rolls on for a gap frame instead (checkpoint_1())."""
        face_x = x2 - self.arena.obstacle_2_size[0] / 2.0
        width, widths, how = task2_vision.bar_width(
            self.bullseyes, self.arrows_2, float(self.p('camera_hfov')), face_x, TASK2_CENTRE_Y, self.side,
            self.arena.obstacle_2_width_range,
            gap_scale=1.0 if self.p('use_sim_time') else float(self.p('bar_gap_scale')))
        return (width, widths, how) if how == 'gap' else (None, [], '')

    def bar_w(self) -> float:
        """The bar's width planned with: measured (up to 5 cm), set up, or the usual one."""
        if self.bar_width is not None:
            return self.bar_width
        if self.known() and self.known()[4] is not None:
            return self.known()[4]
        return self.arena.obstacle_2_size[1]

    def round_bar_width(self, width: float) -> float:
        """UP to the route table's 5 cm step: never planned narrower than measured - 0-5 cm
        more room at the bar's ends (nearest until 2026-10-16, for side walls a fixed
        distance from the bar's real ends - none stand near it on the course)."""
        return round(math.ceil(width / task2_routes.STEP_M - 1e-6) * task2_routes.STEP_M, 3)

    # --------------------------------------------------- laptop planning ----

    def remote_leg(self, start, target, costmap_spec):
        """One leg planned on the laptop (task1_planner): the path, [] for no path, or
        None when it does not answer - then the caller plans it here."""
        self._remote_n += 1
        gen = f'task2:{self._plan_gen}:{self._remote_n}'
        wait = {'ack': threading.Event(), 'done': threading.Event(), 'path': None}
        self._leg_waits[gen] = wait
        self.plan_request_pub.publish(String(data=json.dumps({
            'gen': gen, 'task2': costmap_spec, 'start': list(start), 'checkpoints': [list(target)],
            'params': planner_params.to_flat(planner_params.ACTIVE)})))
        timeout = float(self.p('remote_plan_timeout'))
        try:
            if not wait['ack'].wait(timeout):
                return None
            if not wait['done'].wait(timeout + planner_params.ACTIVE.max_planning_time):
                return None
            return wait['path']
        finally:
            self._leg_waits.pop(gen, None)

    def plan_result_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        wait = self._leg_waits.get(data.get('gen'))
        if wait is None:
            return
        if data.get('ack'):
            wait['ack'].set()
        else:
            wait['path'] = [tuple(p) for p in data.get('path', [])]
            wait['done'].set()

    def costmap_spec(self, x1, x2):
        """What the laptop needs to build the same costmap (task1_planner)."""
        return {'x1': x1, 'x2': x2, 'bar_width': self.bar_w(), 'padding': float(self.p('obstacle_padding')),
                'wall_padding': float(self.p('wall_padding')),
                'obstacle_1_padding': float(self.p('obstacle_1_padding')),
                'layout': os.path.basename(self.layout),
                'arena': task2_routes.arena_to_dict(self.arena)}

    def hybrid_leg(self, costmap, start, target, obstacles=None):
        """Hybrid A* for a leg no clean shape fits: on the laptop if there is one, else here.
        obstacles: (x1, x2) the costmap was built from - preplan() passes its own:
        before START self.x1 is still None, and the laptop's planner crashed on it, so
        every leg waited out the timeout and was planned here (car 2026-10-15)."""
        if self.remote_planner:
            x1, x2 = obstacles or (self.x1, self.x2)
            path = self.remote_leg(start, target, self.costmap_spec(x1, x2))
            if path is not None:
                return path
            self.get_logger().warn("PLAN      the laptop did not answer - planning here")
        return plan_leg(costmap, start, target, progress_callback=self.publish_search_progress)

    def ultrasonic_offset(self) -> float:
        """How far ahead of base_link the ultrasonic sits (URDF, via TF)."""
        if self.us_offset is None:
            try:
                tf = self.tf_buffer.lookup_transform('base_link', 'ultrasonic_link', rclpy.time.Time())
                self.us_offset = tf.transform.translation.x
            except tf2_ros.TransformException:
                self.get_logger().warn("PLAN      no base_link -> ultrasonic_link transform - assuming 0.212 m (URDF)",
                                       throttle_duration_sec=5.0)
                return 0.212
        return self.us_offset

    def sensors(self):
        """Where the IRs are on the car (x, y, yaw on base_link), from the URDF."""
        if len(self.ir_sensors) < 2:
            for name in ('ir', 'ir2'):
                try:
                    t = self.tf_buffer.lookup_transform('base_link', f'{name}_link', rclpy.time.Time()).transform
                    self.ir_sensors[name] = (t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation))
                except tf2_ros.TransformException:
                    pass
        return self.ir_sensors

    def watch_face(self, label, x0, x1, face_y, out, at, use_along=True):
        """Fix the position with whichever IR looks at this face when the car is at `at`."""
        if not self.p('ir_fix'):         # navigation.yaml task2_runner ir_fix (live)
            return
        for name, sensor in self.sensors().items():
            fp = FacePass(label, x0, x1, face_y, out, sensor, use_along)
            if fp.beam(at) is not None:
                self.passes.append((name, fp))
                return

    def watch_obstacle_1(self):
        h = self.arena.obstacle_1_size[1] / 2.0
        d = self.arena.obstacle_1_size[0] / 2.0
        self.watch_face("obstacle 1's side", self.x1 - d, self.x1 + d, TASK2_CENTRE_Y + self.side * h,
                        self.side, (self.x1, TASK2_CENTRE_Y + self.side * float(self.p('lane_offset')), 0.0))

    def watch_bar_ends(self):
        """The bar's end on arrow 2's side (passed heading out) and its other end (heading home).
        'out' only: their 'along' was +4.5..+7.6 cm off in 3 of 4 sim runs (2026-10-10)."""
        h = self.bar_w() / 2.0
        d = self.arena.obstacle_2_size[0] / 2.0
        s2 = SIDE[self.arrows[1]]
        for k, wp in enumerate(self.waypoints[:2]):
            out = s2 if k == 0 else -s2
            self.watch_face(f"the bar's {'first' if k == 0 else 'second'} end", self.x2 - d, self.x2 + d,
                            TASK2_CENTRE_Y + out * h, out, wp, use_along=False)

    def ir_callback(self, name, msg: Range):
        if not self.passes or self.last_odom is None or not math.isfinite(msg.range):
            return
        v = self.last_odom.twist.twist.linear.x
        # The pose where the car was at the reading's time: the EKF pose moved on by
        # the time between their stamps (sim 2026-10-10: without it the 'along' fixes
        # were +2..+10 cm on perfect odometry; a fixed pose_latency x v overshot, -4 cm).
        dt = (rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
              - rclpy.time.Time.from_msg(self.last_odom.header.stamp).nanoseconds)
        x, y, yaw = self.current_pose
        lag = v * max(0.0, min(0.2, dt * 1e-9))
        pose = (x + lag * math.cos(yaw), y + lag * math.sin(yaw), yaw)
        for item in list(self.passes):
            n, fp = item
            if n != name:
                continue
            fp.on_reading(float(msg.range), pose, abs(v))
            bm = fp.beam(pose)
            past = bm is None or not (fp.x0 - 0.10 <= bm[1] <= fp.x1 + 0.10)   # the pose may be cm off
            if fp.seen and past:
                self.passes.remove(item)
                dx, dy, note = fp.correction()
                if math.hypot(dx, dy) >= 0.005:
                    self.shift_pose(dx, dy)
                    self.get_logger().info(f"IR FIX    {fp.label}: {note} -> {self.fmt(self.current_pose)}")
                else:
                    self.get_logger().info(f"IR FIX    {fp.label}: none - {note}")

    def fresh_range(self, max_age: float = 0.3):
        if self.us_range and self.now() - self.us_range[1] <= max_age:
            return self.us_range[0]
        return None

    def ahead_x(self, reading: float, depth: float) -> float:
        """Centre (along the course) of a block `depth` deep whose face (across the
        course) is `reading` from the ultrasonic. The beam runs along the car's
        heading, so an angled reading is foreshortened by cos(heading)."""
        x, _, yaw = self.current_pose
        return x + math.cos(yaw) * (self.ultrasonic_offset() + reading) + depth / 2.0

    def known(self):
        """(x1, x2, arrow 1, arrow 2) from `pixi run setup`, or None (measure them)."""
        return self.known_layout

    def arrow(self, i: int):
        if self.known():
            return self.known()[2 + i]
        counts = self.arrow_counts[i]
        return counts.most_common(1)[0][0] if counts else None

    def need_arrow(self, i: int, guess: str = None) -> bool:
        """Arrow i known -> True. Not yet: stop, wait up to ARROW_WAIT_S, then guess
        (`guess`, else LEFT for arrow 1, RIGHT for arrow 2)."""
        if self.arrows[i] is None:
            self.arrows[i] = self.arrow(i)
        if self.arrows[i] is not None:
            return True
        self.send_cmd(0.0, 0.0)
        if self._arrow_wait_since is None:
            self._arrow_wait_since = self.now()
        if self.now() - self._arrow_wait_since < ARROW_WAIT_S:
            self.get_logger().warn(f"ARROW {i + 1}   not read yet - waiting", throttle_duration_sec=1.0)
            return False
        self.arrows[i] = guess or ('LEFT' if i == 0 else 'RIGHT')
        self._arrow_wait_since = None
        self.get_logger().warn(f"ARROW {i + 1}   never read - guessing {self.arrows[i]}")
        return True

    # ---------------------------------------------------------- callbacks ----

    def ultrasonic_callback(self, msg: Range):
        if math.isfinite(msg.range) and msg.min_range <= msg.range <= msg.max_range:
            self.us_range = (float(msg.range), self.now())

    def yolo_callback(self, msg: String):
        """/yolo_result (one label a frame): only while no box lists arrive - an older
        YOLO node. Then arrow 1 until the car turns out, arrow 2 once it is level
        with obstacle 1 (its face out of view)."""
        if self.boxes_seen_at is not None and self.now() - self.boxes_seen_at < 2.0:
            return
        side = ARROWS.get(msg.data.strip().upper())
        if side is None:
            return
        if self.state in (State.WAITING_FOR_START, State.APPROACH_1):
            self.count_arrow(0, side)
        elif self.state == State.CHECKPOINT_1 or (self.state == State.TO_CHECKPOINT_1 and self.level_with_1()):
            self.count_arrow(1, side)

    def start_run_callback(self, request, response):
        response.success = self.state == State.WAITING_FOR_START and not self.start_received and self.have_pose
        if not self.have_pose:
            response.message = "Not ready: no pose from the EKF yet."
            self.get_logger().warn(f"`start` rejected: {response.message}")
            return response
        if response.success and self.known() and not (self.pre and self.pre.get('legs')):
            # Without the plan START would drive APPROACH_1: straight on until the ultrasonic sees obstacle 1.
            response.success, response.message = False, "Not ready: the setup layout's 4 legs are not planned yet."
            self.get_logger().warn(f"`start` rejected: {response.message}")
            return response
        if response.success:
            self.start_received = True
            self.reset_done = False          # the car is leaving the start
            response.message = "Run started - task 2" + (" (setup layout)." if self.known() else ".")
        else:
            response.message = f"Ignored: {self.state.name} - reset (pixi run reset) to run again."
            self.get_logger().warn(f"`start` rejected: {response.message}")
        return response

    def step_callback(self, request, response):
        """/step (step:=true): before the run = START; standing at a checkpoint = go on
        (the next leg, or at checkpoint 1 with something missing: look again)."""
        if not self.step:
            response.success, response.message = False, "Not in step mode - launch with step:=true."
            return response
        if self.state == State.WAITING_FOR_START:
            return self.start_run_callback(request, response)
        if self.state == State.CHECKPOINT_1 or (self.state == State.FOLLOW and not self.follower.active):
            self.step_go = True
            response.success, response.message = True, "Step."
            return response
        response.success = False
        response.message = (f"Ignored: {self.state.name} - still driving." if self.state not in
                            (State.FINISHED, State.STOPPED) else f"{self.state.name} - reset to run again.")
        return response

    def setup_obstacles_callback(self, request, response):
        """/setup_obstacles without a laptop: this machine's task2 layout on /task2_setup."""
        try:
            data = obstacle_layout.task2_setup_string(self.layout)
        except (OSError, ValueError, KeyError) as exc:
            response.success, response.message = False, f'no task 2 layout in {self.layout} ({exc})'
            self.get_logger().warn(f"SETUP     {response.message}")
            return response
        self.task2_setup_pub.publish(String(data=data))
        response.success, response.message = True, f'task 2 layout from {self.layout}'
        return response

    def task2_setup_callback(self, msg: String):
        """Both obstacles and arrows known (pixi run setup): no measuring this run."""
        if self.state not in (State.WAITING_FOR_START, State.FINISHED, State.STOPPED):
            self.get_logger().warn(f"SETUP     ignored during the run ({self.state.name})")
            return
        try:
            known = obstacle_layout.from_task2_setup(msg.data)
        except (ValueError, KeyError) as exc:
            self.get_logger().warn(f"SETUP     bad /task2_setup {msg.data!r}: {exc}")
            return
        if known == self.known_layout:
            return
        self.known_layout = known
        self.pre = None                  # plan checkpoint 1 again from the known layout
        self.x1_seen = None              # show_obstacle_1() draws it, the bar too (publish_map)
        x1, x2, a1, a2, width = known
        width_note = f", {width * 100:.0f} cm wide" if width is not None else ""
        self.get_logger().info(f"SETUP     known: obstacle 1 at ({markers.cell(x1)},{markers.cell(TASK2_CENTRE_Y)}) "
                               f"{a1}, bar at ({markers.cell(x2)},{markers.cell(TASK2_CENTRE_Y)}) {a2}{width_note}"
                               f" - no ultrasonic / YOLO needed")

    def stop_run_callback(self, request, response):
        self._plan_gen += 1
        self.set_path([])
        self.set_state(State.STOPPED)
        self.send_cmd(0.0, 0.0)
        self.end_run()
        self.get_logger().info(f"E-STOP    {self.run_time():.1f} s")
        response.success, response.message = True, "Stopped."
        return response

    def set_pose_callback(self, msg):
        """A pose reset (car back in the carpark) readies the next run; during a
        run it stops it first."""
        if self.state not in (State.FINISHED, State.STOPPED, State.WAITING_FOR_START):
            self.send_cmd(0.0, 0.0)
            self.get_logger().warn("RESET     during a run - stopping it")
        self.reset_run()
        self.reset_done = True
        self._ready_note = None          # READY again after a reset
        self.publish_map()
        if self.pre:                     # the setup layout's plan, kept: show it again
            self.waypoints = list(self.pre.get('waypoints', []))
            self.path_pub.publish(markers.route_path([self.pre['path']] + self.pre.get('legs', []), self.stamp()))
            self.publish_waypoints()
        self.get_logger().info("RESET     ready - waiting for START" + (" (planned)" if self.pre else ""))

    # -------------------------------------------------------------- control ----

    # -------------------------------------------------------- tablet lines ----

    def plan_state(self) -> str:
        """PLAN: as task 1's - DONE once the run can START (the setup layout: all 4 legs;
        measuring: the path to checkpoint 1), PLANNING while that is being planned."""
        planning = self.leg_paths and any(p is None for p in self.leg_paths)
        if planning or (self.pre and self.pre.get('planning')):
            return 'PLANNING'
        if self.pre and self.pre.get('path') and (self.pre.get('legs') or not self.known()):
            return 'DONE'
        return 'WAITING'

    def status_text(self) -> str:
        """STATUS: as task 1's: Waiting / Ready / Going to checkpoint n / Finished / Stopped."""
        if self.state == State.STOPPED:
            return 'Stopped'
        if self.state == State.FINISHED:
            return 'Finished'
        if self.state == State.WAITING_FOR_START:
            return 'Ready' if self.plan_state() == 'DONE' and self.reset_done else 'Waiting'
        if self.state == State.FOLLOW and self.leg < len(self.waypoints):
            return f'Going to checkpoint {self.leg + 2}'
        return 'Going to checkpoint 1'

    def sync_indicators(self):
        """PLAN / RESET / STATUS to the tablet when they change, and every 2 s (a bridge
        that came up later) - the same lines as task 1."""
        now = self.now()
        if now - self._last_heartbeat >= 2.0:
            self._last_heartbeat = now
            self._last_sent.clear()
        lines = {'PLAN': f'PLAN:{self.plan_state()}',
                 'RESET': f'RESET:{"DONE" if self.reset_done else "WAITING"}',
                 'STATUS': f'STATUS:{self.status_text()}'}
        for key, line in lines.items():
            if self._last_sent.get(key) != line:
                self._last_sent[key] = line
                self.send_bt(line)

    def control_loop(self):
        self.sync_indicators()
        handler = {
            State.WAITING_FOR_START: self.waiting,
            State.TO_CHECKPOINT_1: self.to_checkpoint_1,
            State.APPROACH_1: self.approach_1,
            State.SWERVE_OUT: self.swerve_out,
            State.SWERVE_BACK: self.swerve_back,
            State.CHECKPOINT_1: self.checkpoint_1,
            State.FOLLOW: self.follow,
        }.get(self.state)
        if handler:
            handler()
        else:
            self.send_cmd(0.0, 0.0)

    def waiting(self):
        self.send_cmd(0.0, 0.0)
        if not self.start_received:
            self.show_obstacle_1()
            self.live_home()
            self.preplan()
            return
        self.run_start = self.now()
        pre = self.pre
        if pre and pre.get('path'):
            # Planned at home: straight to the waypoint beside obstacle 1.
            self.arrows[0], self.x1, self.side = pre['arrow'], pre['x1'], SIDE[pre['arrow']]
            self.lane_waypoint, self.cp1_planned = pre['goal'], True
            self.set_path(pre['path'])
            self.watch_obstacle_1()
            self.get_logger().info("START")
            self.snapshot(1, self._arrow1_stamp, f"OBSTACLE 1 -> {pre['arrow']}")
            self.publish_map()
            self.set_state(State.TO_CHECKPOINT_1)
            self.publish_waypoints()
            return
        self.get_logger().info("START     arrow 1 not read yet - driving on until it is")
        self.set_state(State.APPROACH_1)

    def live_home(self):
        """step:=true at home: arrow 1 from the last LIVE_REFRESH_S of reads only, so a
        card turned round shows within one refresh (preplan() re-plans when arrow 1 or
        obstacle 1 changes). The setup layout: nothing to read."""
        if not self.step or self.known():
            return
        now = self.now()
        if self._home_look_since is None:
            self._home_look_since = now
        if now - self._home_look_since < LIVE_REFRESH_S:
            return
        self._home_look_since = now
        counts = self.arrow_counts[0]
        if counts:
            # Keep this window's winner a single read, so arrow 1 is never unknown
            # in between; the next window's reads outvote it if the card turned.
            self.arrow_counts[0] = Counter({counts.most_common(1)[0][0]: 1})

    def home_x1(self):
        """Obstacle 1's centre as seen from home (the ultrasonic, or the setup layout), or None."""
        if self.known():
            return self.known()[0]
        r = self.fresh_range()
        if r is None or not (US_VALID[0] <= r <= US_VALID[1]):
            return None
        return self.ahead_x(r, self.arena.obstacle_1_size[0])

    def show_obstacle_1(self):
        """At home: draw obstacle 1 and its costmap as soon as the ultrasonic reads it."""
        x1 = self.home_x1()
        if x1 is None:
            return
        if self.x1_seen is None or abs(self.x1_seen - x1) > 0.03:
            self.x1_seen = x1
            self.costmap = self.make_costmap(x1, None)
            self.publish_map()

    def preplan(self):
        """Before GO: once obstacle 1 and arrow 1 are known (setup, or the ultrasonic
        and YOLO), plan start -> beside obstacle 1 (again if either changes). With the
        setup layout obstacle 2 is known too: the legs to checkpoints 2, 3, 4 as well,
        all before START - planned at checkpoint 1 at 0.9 m/s, the Hybrid A* leg came too
        late on the busy Pi and the car rolled 0.5 m into the bar (car 2026-10-14)."""
        arrow, x1 = self.arrow(0), self.home_x1()
        if arrow is None or x1 is None:
            return
        known, pre = self.known(), self.pre
        # A failed plan is tried again, a found one only if the layout changes.
        done = pre and pre['path'] and (pre.get('legs') or not known)
        if pre and (pre.get('planning') or (done and pre['arrow'] == arrow and abs(pre['x1'] - x1) < 0.03)):
            return
        goal = task2_routes.checkpoint_1(x1, arrow, float(self.p('lane_offset')), float(self.p('cp1_back')))
        self.pre = pre = {'arrow': arrow, 'x1': x1, 'goal': goal, 'path': None, 'planning': True}
        costmap = self.make_costmap(x1, None)

        # From the start pose, like task 1 - not the live pose: just after a reset that is
        # (0, 0) outside the course, and Hybrid A* searched max_planning_time (60 s) for
        # nothing (car 2026-10-14, with setup).
        def plan(start=self.start):
            t0 = time.monotonic()
            try:
                goal, path = self.cp1_lined_up(start, x1, arrow, costmap)
                pre['goal'] = goal
                pre['path'] = path or self.hybrid_leg(costmap, start, goal, (x1, None))
                if not pre['path']:
                    self.ready_line("PLAN      no path to CP1 - trying again", warn=True)
                    return
                self.lane_waypoint = goal
                self.path_pub.publish(markers.route_path([pre['path']], self.stamp()))
                self.publish_waypoints()
                if not known:
                    back, want = x1 - goal[0], float(self.p('cp1_back'))
                    d1 = x1 - self.arena.obstacle_1_size[0] / 2.0 - self.arena.opening_x
                    self.ready_line(f"READY     arrow 1 {arrow}, obstacle 1 at {d1 * 100:.0f} cm"
                                    + (f", CP1 {back * 100:.0f} cm back (no room for {want * 100:.0f})"
                                       if back < want - 0.01 else ''))
                    return
                x2, arrow2 = known[1], known[3]
                waypoints = self.bar_waypoints(x2, arrow2)
                full = self.make_costmap(x1, x2)
                legs, pose = [], goal
                for idx, target in enumerate(waypoints):
                    path = self.geometric_leg(idx, pose, target, full) or self.hybrid_leg(full, pose, target, (x1, x2))
                    if not path:
                        self.ready_line(f"PLAN      no path to CP{idx + 2} - trying again", warn=True)
                        return
                    legs.append(path)
                    pose = target
                pre.update(x2=x2, arrow2=arrow2, waypoints=waypoints, legs=legs)
                self.waypoints = waypoints
                self.path_pub.publish(markers.route_path([pre['path']] + legs, self.stamp()))
                self.publish_waypoints()
                self.ready_line(f"READY     setup layout: arrows {arrow} / {arrow2}, all 4 legs planned")
            except Exception:
                self.get_logger().error(f"PLAN      crashed:\n{traceback.format_exc()}")
            finally:
                pre['planning'] = False
        threading.Thread(target=plan, daemon=True).start()

    def shapes(self) -> task2_paths.Shapes:
        """The clean shapes' turning circles and speed profile (live settings)."""
        return task2_paths.Shapes.from_params(self.car, float(self.p('straight_speed')),
                                              float(self.p('max_lateral_accel')))

    def geometric_leg(self, idx, start, target, costmap):
        """Leg idx as clean geometry, or None: 1 -> 2 an S, straight into
        checkpoint 2; 2 -> 3 half a circle round the bar; 3 -> 4 an S onto the
        centre line, straight into the carpark (task2_paths.py)."""
        if idx == 1:
            return task2_paths.round_path(start, target, costmap, self.shapes())
        if idx == 0:
            straight, margin = float(self.p('line_up_dist')), 0.04
        else:
            straight, margin = float(self.p('home_straight_dist')), float(self.p('home_margin'))
        return task2_paths.stepped_lined_up_path(start, target, costmap, self.shapes(), straight, margin)

    def geometric_legs(self, start, waypoints, costmap):
        paths, pose = [], start
        for idx, target in enumerate(waypoints):
            paths.append(self.geometric_leg(idx, pose, target, costmap))
            pose = target
        return paths

    def cp1_lined_up(self, start, x1, arrow, costmap):
        """(checkpoint 1, the lined-up path to it or None): as far back as cp1_back
        behind obstacle 1's centre (the camera in front of it) WITH at least
        cp1_min_straight of straight before it, else 10 cm nearer at a time. The
        furthest-back fit had none on every course - the car reached checkpoint 1
        still turning and stopped 23 deg in (bag 2026-10-16 03:36). Nothing with the
        straight: the furthest back that fits at all, as before. Leg 1's table route
        follows (table_route() - routes for every 10 cm back)."""
        lane, want = float(self.p('lane_offset')), float(self.p('cp1_back'))
        backs, b = [], want
        while b > -1e-6:
            backs.append(round(max(0.0, b), 3))
            b -= 0.10
        for min_straight in (float(self.p('cp1_min_straight')), 0.0):
            for back in backs:
                goal = task2_routes.checkpoint_1(x1, arrow, lane, back)
                path = self.cp1_path(start, goal, costmap, min_straight)
                if path:
                    return goal, path
        return task2_routes.checkpoint_1(x1, arrow, lane, 0.0), None

    def cp1_path(self, start, goal, costmap, min_straight: float):
        """The lined-up S into checkpoint 1 with at least min_straight before it
        (task2_paths.LADDER's straights and margins, smooth before arc)."""
        straight = float(self.p('line_up_dist'))
        for shape in ('smooth', 'arc'):
            for f_straight, f_margin in task2_paths.LADDER:
                if straight * f_straight < min_straight - 1e-6:
                    continue
                path = task2_paths.lined_up_path(start, goal, costmap, self.shapes(), straight * f_straight,
                                                 0.04 * f_margin, shape)
                if path:
                    return path
        return None

    def lined_up_path(self, start, goal, costmap, straight, margin=0.04):
        """start -> goal (same heading), straight again `straight` before it (task2_paths.py)."""
        return task2_paths.stepped_lined_up_path(start, goal, costmap, self.shapes(), straight, margin)

    def level_with_1(self, max_yaw: float = math.radians(8.0)) -> bool:
        """In the lane beside obstacle 1 and pointing (nearly) straight: the ultrasonic
        and the camera now look past it at obstacle 2."""
        if self.x1 is None:
            return False
        # From the straight into checkpoint 1 on: cp1_back 0.4 put checkpoint 1 before
        # x1 - LEVEL_X_M, and the ultrasonic's readings of the bar there (1.0-1.9 m) were
        # thrown away - the camera's guess, 10-14 cm short, made the bar too narrow (bags
        # 2026-10-15).
        cp1_x = self.lane_waypoint[0] if self.lane_waypoint else self.x1 - float(self.p('cp1_back'))
        start = min(self.x1 - LEVEL_X_M, cp1_x - float(self.p('line_up_dist')))
        return self.current_pose[0] >= start and abs(self.current_pose[2]) < max_yaw

    def read_obstacle_2(self):
        """One ultrasonic reading of obstacle 2 (its centre along the course), kept
        if it is beyond obstacle 1."""
        if self.known():
            self.lane_readings = [self.known()[1]]
            return
        r = self.fresh_range(0.1)
        if r is not None and US_VALID[0] <= r <= US_VALID[1] and self.level_with_1(math.radians(11.0)):
            x2 = self.ahead_x(r, self.arena.obstacle_2_size[0])
            if x2 > self.x1 + 0.3:
                self.lane_readings.append(x2)

    def to_checkpoint_1(self):
        """The path from the carpark to checkpoint 1; obstacle 2 is measured in its
        last moments (pointing straight, level with obstacle 1)."""
        self.leg_pub.publish(markers.leg_markers(self.follower.path, self.follower.last_target,
                                                 self.current_pose, self.stamp()))
        self.read_obstacle_2()
        self.check_stuck()
        if not self.follower_step(stop_at_end=self.step):
            return
        # Keep rolling straight on while checkpoints 2-4 are planned (no stop).
        if not self.step:
            self.hold_heading(max(abs(self._last_cmd[0]), float(self.p('path_speed'))), 0.0)
        self.arrive_checkpoint_1()

    def arrive_checkpoint_1(self):
        if self.step:
            self.send_cmd(0.0, 0.0)
            self.set_path([])
            self._look_since = self.now()
            self.get_logger().info(f"CP1       {self.cp1_offness()} - stopped, looking {STEP_LOOK_S:.0f} s")
        else:
            self.get_logger().info(f"CP1       {self.cp1_offness()}")
        self.set_state(State.CHECKPOINT_1)
        self.checkpoint_1()

    def approach_1(self):
        """Fast and straight toward obstacle 1 until arrow 1 is read. Then, like at
        home: checkpoint L beside obstacle 1 and the diagonal to it, from here. Only
        if no diagonal fits any more (too close): the fixed swerve, at the latest at
        swerve_trigger_dist (waiting there for the arrow if it still isn't read)."""
        r = self.fresh_range()
        arrow = self.arrow(0)
        if arrow is not None and r is not None and US_VALID[0] <= r <= US_VALID[1]:
            x1 = self.ahead_x(r, self.arena.obstacle_1_size[0])
            if self.costmap is None or self.x1_seen is None or abs(self.x1_seen - x1) > 0.03:
                self.x1_seen, self.costmap = x1, self.make_costmap(x1, None)
            x, y, _ = self.current_pose
            goal, path = self.cp1_lined_up((x, y, 0.0), x1, arrow, self.costmap)
            if path:
                self.arrows[0], self.x1, self.side, self.lane_waypoint = arrow, x1, SIDE[arrow], goal
                self.cp1_planned = True
                self.set_path(path)
                self.watch_obstacle_1()
                self.get_logger().info(f"ARROW 1   {arrow} (read on the way)")
                self.snapshot(1, self._arrow1_stamp, f"OBSTACLE 1 -> {arrow}")
                self.path_pub.publish(markers.route_path([path], self.stamp()))
                self.publish_map()
                self.set_state(State.TO_CHECKPOINT_1)
                self.publish_waypoints()
                return
        # Not inside the carpark: its side walls leave no room to swerve.
        out = self.current_pose[0] >= self.arena.opening_x
        early = (out and r is not None and r <= float(self.p('swerve_early_dist'))
                 and self.arrow(0) is not None)
        latest = r is not None and r <= float(self.p('swerve_trigger_dist'))
        self.us_trigger_count = self.us_trigger_count + 1 if (early or latest) else 0
        if self.us_trigger_count < 2:
            self.hold_heading(float(self.p('straight_speed')), 0.0)
            return
        if not self.need_arrow(0):
            return
        self.x1 = self.ahead_x(r, self.arena.obstacle_1_size[0])
        self.side = SIDE[self.arrows[0]]
        R = self.car.minimum_turning_radius_left + self.car.minimum_turning_radius_right
        self.swerve_angle = math.acos(max(-1.0, 1.0 - float(self.p('lane_offset')) / R))
        self.get_logger().info(f"SWERVE    {self.arrows[0]}, {math.degrees(self.swerve_angle):.0f} deg each way")
        self.snapshot(1, self._arrow1_stamp, f"OBSTACLE 1 -> {self.arrows[0]}")
        self.publish_map()
        self.set_state(State.SWERVE_OUT)

    def swerve_out(self):
        """Full lock toward the arrow's side until turned swerve_angle."""
        v = float(self.p('path_speed'))
        if self.side * self.current_pose[2] >= self.swerve_angle - ARC_LEAD_RAD:
            self.set_state(State.SWERVE_BACK)
        self.steer(v, self.side * (self.car.steering_limit_left if self.side > 0 else self.car.steering_limit_right))

    def swerve_back(self):
        """Full lock the other way until pointing straight again."""
        v = float(self.p('path_speed'))
        if self.side * self.current_pose[2] <= ARC_LEAD_RAD:
            self.lane_waypoint = (self.current_pose[0], self.current_pose[1], 0.0)
            self.arrive_checkpoint_1()
            return
        self.steer(v, -self.side * (self.car.steering_limit_right if self.side > 0 else self.car.steering_limit_left))

    def checkpoint_1(self):
        """At checkpoint 1: arrow 2 (YOLO), obstacle 2's distance (the ultrasonic, read on
        the straight into it - checked by the camera) and the bar's width (arrow 2 + the
        near bullseye) -> checkpoints 2, 3, 4 and the path through them. All known on
        arrival: on at once, rolling. Something missing (or the bar far): roll on
        straight CP1_ROLL_M, then stop, reading up to CP1_WAIT_S in all - then arrow 2
        still unread: ARROW_WAIT_S more, then the nearer end; no distance: the run
        stops; no gap width: the usual one."""
        if self.pre and self.pre.get('legs'):
            return self.follow_setup_legs()
        if self.step:
            return self.step_checkpoint_1()
        self.read_obstacle_2()
        if self.arrows[1] is None:
            self.arrows[1] = self.arrow(1)
        x2, why, src = self.bar_distance()
        known_width = self.bar_width is not None or (self.known() and self.known()[4] is not None)
        measured = self.measure_bar(x2) if x2 is not None and not known_width else (None, [], '')
        missing = [name for name, gone in (('arrow 2', self.arrows[1] is None), ("obstacle 2's distance", x2 is None),
                                           ('arrow 2 + bullseye', x2 is not None and not known_width
                                            and measured[0] is None))
                   if gone]
        cp1_x = self.lane_waypoint[0] if self.lane_waypoint else self.current_pose[0]
        rolling = self.current_pose[0] - cp1_x < CP1_ROLL_M
        # A far bar's width read low (67 / 69 cm for 76 at 1.2-1.5 m, car 2026-10-16):
        # roll on closer while the roll lasts (bar_width() keeps the nearest frames).
        if rolling and x2 is not None and not known_width and measured[0] is not None \
                and self.bar_gap(x2) > WEAK_WIDTH_DIST_M:
            missing.append('a closer width')
        if missing:
            now = self.now()
            if self._cp1_wait_since is None:
                self._cp1_wait_since = now
            if now - self._cp1_wait_since < CP1_WAIT_S:
                if rolling:
                    self.hold_heading(float(self.p('cp1_speed')), 0.0)
                else:
                    self.send_cmd(0.0, 0.0)
                note = f"{'rolling' if rolling else 'stopped'} - reading {', '.join(missing)}"
                if note != self._cp1_note:
                    self._cp1_note = note
                    self.get_logger().info(f"CP1       {note}")
                return
        if x2 is None:
            self.send_cmd(0.0, 0.0)
            self.get_logger().error("CP1       obstacle 2 not seen (ultrasonic or camera) - stopping")
            self.set_state(State.STOPPED)
            return
        # Still unread: wait for it (then the nearer end) - a guess is the wrong side half
        # the time, a failed run.
        if self.arrows[1] is None and not self.need_arrow(1, 'LEFT' if self.side > 0 else 'RIGHT'):
            return
        if not known_width:
            width, widths, how = measured
            self._bar_info = (width, how)
            self.bar_width = self.round_bar_width(self.safe_width(width, how, x2))
        self.plan_bar(x2, why, src)

    def step_checkpoint_1(self):
        """step:=true at checkpoint 1: a live preview. Stand still collecting arrow 2's
        and the bullseyes' boxes and the ultrasonic - STEP_LOOK_S the first time, then
        LIVE_REFRESH_S each - and from each look alone (fresh: the bar may have been
        moved or swapped) plan checkpoints 2, 3, 4 and their legs again if anything
        changed (preview_bar(), drawn in Foxglove). Anything missing: say what, keep
        the last plan. SPACE commits the latest plan and drives leg 1 (commit_preview());
        with none yet it looks again. No glance, hedge or guess: the car does not move here."""
        self.send_cmd(0.0, 0.0)
        if self.step_go and self._preview:
            return self.commit_preview()
        self.read_obstacle_2()
        if self._look_since is None:
            self.new_look()
        if self.now() - self._look_since < (STEP_LOOK_S if self._looks == 0 else LIVE_REFRESH_S):
            return
        # The first look keeps what the way in saw (arrow 2 glimpsed past obstacle 1). Later
        # ones: boxes of frames taken before the look began (YOLO's latency) are the old layout's.
        if self._looks:
            t0 = self._look_since
            self.bullseyes = [b for b in self.bullseyes if b[0] >= t0]
            self.arrows_2 = [a for a in self.arrows_2 if a[0] >= t0]
        arrow2 = self.arrow(1)
        x2, why, src = self.bar_distance()
        missing, width = [], None
        if arrow2 is None:
            missing.append('arrow 2')
        if x2 is None:
            missing.append("obstacle 2's distance")
        else:
            if self.known() and self.known()[4] is not None:
                width = self.known()[4]
            else:
                raw, widths, how = self.measure_bar(x2)
                if raw is None:
                    missing.append('a bullseye (bar width)')
                else:
                    width = self.round_bar_width(self.safe_width(raw, how, x2))
                    self._bar_info = (raw, how)
        self._looks += 1
        go = self.step_go
        self.step_go = False
        self.new_look()
        if missing:
            note = ', '.join(missing)
            if note != self._step_note:
                self._step_note = note
                self.get_logger().warn(f"LIVE      no {note} - " + ("keeping the last plan" if self._preview else
                                       f"looking again every {LIVE_REFRESH_S:.0f} s"))
            if go:
                self.get_logger().info("STEP      nothing planned yet - looking again")
            return
        self._step_note = None
        self._preview_src = src
        self.preview_bar(x2, why, arrow2, width)
        if go:                                # SPACE came while it looked: this plan goes
            self.step_go = True

    def new_look(self):
        """step:=true at checkpoint 1: start a look from scratch (step_checkpoint_1())."""
        self._look_since = self.now()
        self.lane_readings = []
        self.bullseyes = []
        self.arrows_2 = []
        self.arrow_counts[1] = Counter()
        self.arrows[1] = None

    def preview_bar(self, x2: float, why: str, arrow2: str, width: float):
        """step:=true at checkpoint 1: one look's reading. Changed from the last plan
        (arrow 2, the bar's width, or its distance by 3 cm or more): checkpoints 2, 3, 4
        and their legs planned again from here and drawn; older plans still being
        worked on are dropped (_plan_gen). The car stays."""
        last = self._preview
        if (last and last[2] == arrow2 and last[3] == width and abs(last[0] - x2) < 0.03):
            return
        self._preview = (x2, why, arrow2, width)
        if why:
            self.get_logger().warn(f"LIVE      {why}")
        self.x2, self.bar_width = x2, width
        self.arrows[1] = arrow2              # for the route table and the drawing - cleared again below
        self.waypoints = self.bar_waypoints(x2, arrow2)
        self.build_costmap()
        self.publish_map()
        self.publish_waypoints()
        self._plan_gen += 1
        first = self.table_route()
        self.arrows[1] = None                # arrow 2 is counted afresh each look (count_arrow())
        x, y, _ = self.current_pose
        start = self.lane_waypoint if first else (x, y, 0.0)
        self.leg_paths = [first] + [None] * (len(self.waypoints) - 1)
        self.path_pub.publish(markers.route_path([p for p in self.leg_paths if p], self.stamp()))
        self.get_logger().info(f"LIVE      {self.obstacle_2_text(arrow2, x2, width, self._preview_src)} - SPACE")
        threading.Thread(target=self.plan_legs, daemon=True,
                         args=(start, self._plan_gen, self.leg_paths, list(self.waypoints), self.costmap,
                               list(self.leg_paths))).start()

    def commit_preview(self):
        """step:=true: SPACE at checkpoint 1 - the latest live plan is the run's: no more
        looks; plan_bar() plans it once more (the same reading) and leg 1 goes."""
        x2, why, arrow2, width = self._preview
        self.arrows[1], self.bar_width = arrow2, width
        self._plan_gen += 1                  # a preview's legs still being planned are dropped
        self.get_logger().info("STEP      committed")
        self.plan_bar(x2, '', self._preview_src)                # step_go stays set: follow() drives leg 1 once it is planned

    def plan_bar(self, x2: float, why: str, src: str):
        """Obstacle 2 at x2 (its distance from `src`), arrow 2 and the bar's width known:
        checkpoints 2, 3, 4 and their legs (leg 1 from the route table if it fits), then
        FOLLOW."""
        if why:
            self.get_logger().warn(f"OBSTACLE 2  {why}")
        self.x2 = x2
        cy = TASK2_CENTRE_Y
        self.waypoints = self.bar_waypoints(self.x2, self.arrows[1])
        # Planned live (no table route): from where the car will be once the plan is
        # ready - it rolls on meanwhile (follow(), ROLL_ON_M); passed points are trimmed.
        # Step mode: it stands still, so from right here.
        x, y, _ = self.current_pose
        stopped = self.step or abs(self._last_cmd[0]) < 0.05
        start = (x, y, 0.0) if stopped else (x + abs(self._last_cmd[0]) * 0.15 + LIVE_PLAN_LEAD_M, y, 0.0)
        text = self.obstacle_2_text(self.arrows[1], x2, self.bar_w(), src)
        self.get_logger().info(f"OBSTACLE 2  {text}")
        self.snapshot(2, self.width_frame_stamp(), f"OBSTACLE 2 -> {text}")
        self.build_costmap()
        self.publish_map()
        self.watch_bar_ends()
        self.leg, self.plan_start = 0, start
        self.set_state(State.FOLLOW)
        self.publish_waypoints()
        # Checkpoint 1 -> 2 from the route table (planned offline, instant), from the
        # planned checkpoint 1 - the car has just passed it, rolling. Legs 2 and 3 then
        # in the background while it drives leg 1: the home S's search took up to
        # ~0.2 s on a laptop (several times that on the Pi) - not in the control loop.
        first = self.table_route()
        if first:
            start = self.lane_waypoint
        self.leg_paths = [first] + [None] * (len(self.waypoints) - 1)
        if first:
            self.path_pub.publish(markers.route_path([first], self.stamp()))
        threading.Thread(target=self.plan_legs, daemon=True,
                         args=(start, self._plan_gen, self.leg_paths, list(self.waypoints), self.costmap,
                               list(self.leg_paths))).start()

    def bar_x_from_camera(self):
        """The bar's centre (along the course) from the bullseyes' sizes, or None: only
        those well beyond obstacle 1 (d2 >= 0.6 m) and not past the longest course."""
        hfov = float(self.p('camera_hfov'))
        far1 = self.x1 + self.arena.obstacle_1_size[0] / 2.0
        faces = []
        for _, box, w, cam in self.bullseyes:
            if not task2_vision.confident(box):
                continue
            bx, _ = task2_vision.bullseye_position(box, w, hfov, cam)
            if far1 + task2_routes.D2_RANGE[0] - 0.15 <= bx <= far1 + task2_routes.D2_RANGE[1] + 0.3:
                faces.append(bx)
        face = task2_vision.median(faces)
        return None if face is None else face + self.arena.obstacle_2_size[0] / 2.0

    def choose_bar_x(self, x2_us, x2_cam):
        """The ultrasonic's distance to the bar - more precise - unless the camera's
        disagrees by more than BAR_X_AGREE_M. Then whichever the bullseyes agree with:
        at the right distance their sizes match their directions (task2_vision
        .bar_half_width's check), at a wrong one they don't. The ultrasonic misses a
        bar narrower than the lane is wide - it looks straight past the end (sim
        2026-10-14: 25 cm too far); the camera can be fooled by false matches. (x2, note)."""
        if x2_cam is None:
            return x2_us, ''
        if x2_us is None:
            return x2_cam, f"from the camera only ({markers.cell(x2_cam)}) - no ultrasonic reading"
        if abs(x2_us - x2_cam) <= BAR_X_AGREE_M:
            return x2_us, ''
        n_us, n_cam = self.bullseyes_fitting(x2_us), self.bullseyes_fitting(x2_cam)
        if n_cam > n_us:
            return x2_cam, (f"ultrasonic {x2_us - x2_cam:+.2f} m off the camera's, the bullseyes fit the camera's "
                            f"({n_cam} / {n_us}) - it missed the bar, using the camera's")
        return x2_us, (f"camera {x2_cam - x2_us:+.2f} m off the ultrasonic's, the bullseyes fit the "
                       f"ultrasonic's ({n_us} / {n_cam}) - keeping it")

    def bullseyes_fitting(self, x2: float) -> int:
        """How many confident bullseyes fit a bar face at x2 (size vs direction)."""
        face_x = x2 - self.arena.obstacle_2_size[0] / 2.0
        hfov = float(self.p('camera_hfov'))
        return sum(1 for _, box, w, cam in self.bullseyes if task2_vision.confident(box)
                   and task2_vision.bar_half_width(box, w, hfov, cam, face_x, TASK2_CENTRE_Y) is not None)

    def safe_width(self, width, how: str, x2: float) -> float:
        """The bar's width to plan with: the measured one (gap method), but at least the
        usual width if it was still read from beyond WEAK_WIDTH_DIST_M after the roll - far
        readings came out low (car 2026-10-16: 67 / 69 for 76 at 1.2-1.5 m; checkpoint 2
        too close, it touched the bar). Wider only moves the path round the bar out."""
        usual = self.arena.obstacle_2_size[1]
        if width is None:
            return usual
        return max(width, usual) if (how != 'gap' or self.bar_gap(x2) > WEAK_WIDTH_DIST_M) else width

    def bar_gap(self, x2: float) -> float:
        """m from the ultrasonic to the bar's face now."""
        return x2 - self.arena.obstacle_2_size[0] / 2.0 - (self.current_pose[0] + self.ultrasonic_offset())

    def ready_line(self, text: str, warn: bool = False):
        """A READY / plan-failure line before START - only when it says something new
        (the plan to checkpoint 1 is re-made on every reading change)."""
        if text != self._ready_note:
            self._ready_note = text
            # Two call sites: rclpy refuses one call site logging at two severities.
            if warn:
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)

    def cp1_offness(self) -> str:
        """'2 deg, +3 cm off the lane' at checkpoint 1."""
        lane = self.lane_waypoint[1] if self.lane_waypoint else self.current_pose[1]
        return (f"{math.degrees(self.current_pose[2]):+.0f} deg, "
                f"{(self.current_pose[1] - lane) * 100:+.0f} cm off the lane")

    def bar_distance(self):
        """(obstacle 2's centre x or None, a disagreement note, 'ultrasonic' / 'camera')."""
        x2_us = statistics.median(self.lane_readings) if self.lane_readings else None
        x2_cam = self.bar_x_from_camera()
        if x2_us is None and x2_cam is None:
            return None, '', ''
        if x2_us is None:
            return x2_cam, '', 'camera'
        x2, why = self.choose_bar_x(x2_us, x2_cam)
        return x2, why, 'ultrasonic' if x2 == x2_us else 'camera'

    def obstacle_2_text(self, arrow2, x2: float, width: float, src: str) -> str:
        """'arrow LEFT, 47 cm -> 45 (gap), 1.02 m (ultrasonic)': the bar's distance from the
        ultrasonic's face."""
        raw, how = self._bar_info
        raised = raw is not None and width > self.round_bar_width(raw) + 1e-6
        w = (f"{raw * 100:.0f} cm -> {width * 100:.0f} ({how}{', weak - at least usual' if raised else ''})"
             if raw is not None
             else f"{width * 100:.0f} cm ({'setup' if how == 'setup' else 'usual - no bullseye'})")
        face = x2 - self.arena.obstacle_2_size[0] / 2.0
        d = face - (self.current_pose[0] + self.ultrasonic_offset())
        return f"arrow {arrow2}, {w}, {d:.2f} m ({src or 'setup'})"

    def width_frame_stamp(self) -> float:
        """The newest frame with arrow 2 and a bullseye - where the width came from - for
        the snapshot (else the newest with either; 0: the newest frame)."""
        bulls = {b[0] for b in self.bullseyes}
        pairs = [t for t, _ in self.arrows_2 if any(abs(t - b) <= 0.02 for b in bulls)]
        stamps = pairs or list(bulls) + [t for t, _ in self.arrows_2]
        return max(stamps) if stamps else 0.0

    def snapshot(self, obstacle: int, stamp: float, caption: str):
        """Ask scan_snapshots (the laptop) to save the annotated frame `stamp`."""
        self.snap_pub.publish(String(data=json.dumps({'obstacle': obstacle, 'stamp': stamp, 'caption': caption})))

    def table_route(self):
        """Checkpoint 1 -> 2 from the route table, in the map frame, or None (then
        planned live): only from the planned checkpoint 1 (not a swerve's end), with
        the car still near it and the table built for these settings."""
        if not (self.cp1_planned and self.lane_waypoint and self.routes_ok()):
            self._leg1_note = 'LIVE (no table route from here)'
            return None
        cp1 = self.lane_waypoint
        x, y, _ = self.current_pose
        if abs(y - cp1[1]) > 0.06 or not (CP1_WINDOW_M[0] < x - cp1[0] < CP1_WINDOW_M[1]):
            self._leg1_note = f"LIVE (car {(x - cp1[0]) * 100:+.0f} / {(y - cp1[1]) * 100:+.0f} cm off CP1)"
            return None
        d2 = (self.x2 - self.arena.obstacle_2_size[0] / 2.0) - (self.x1 + self.arena.obstacle_1_size[0] / 2.0)
        # Checkpoint 1's own distance behind obstacle 1: cp1_back, or nearer on a short
        # course (cp1_lined_up()) - the table has routes for each (task2_routes.cp1_backs()).
        local, why = self.routes.lookup(d2, self.bar_w(), self.arrows[0], self.arrows[1], self.x1 - cp1[0])
        if local is None:
            self._leg1_note = f"LIVE ({why})"
            return None
        self._leg1_note = 'table'
        return task2_paths.to_world(cp1, [tuple(p) for p in local])

    def bar_waypoints(self, x2, arrow2):
        """Checkpoints 2 (the bar's end on arrow 2's side), 3 (its other end, heading
        back) and 4 (home)."""
        return task2_routes.bar_checkpoints(x2, arrow2, self.bar_w(), float(self.p('side_clearance'))) + [self.home]

    def follow_setup_legs(self):
        """At checkpoint 1 with the setup layout: checkpoints 2-4 and their legs were
        all planned before START (preplan) - nothing to compute while rolling."""
        x2, arrow2 = self.pre['x2'], self.pre['arrow2']
        self.x2, self.arrows[1] = x2, arrow2
        self.waypoints = list(self.pre['waypoints'])
        self.leg_paths = list(self.pre['legs'])
        self.get_logger().info(f"OBSTACLE 2  arrow {arrow2} (setup layout, legs planned before START)")
        self.build_costmap()
        self.publish_map()
        self.watch_bar_ends()
        self.leg, self.plan_start = 0, self.pre['goal']
        self.set_state(State.FOLLOW)
        self.publish_waypoints()
        self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))

    def follow(self):
        """Drive the planned legs; until leg 1 is ready, keep straight to its start."""
        if not self.follower.active:
            path = self.leg_paths[self.leg] if self.leg < len(self.leg_paths) else []
            if path is None:                   # still planning
                # Leg 1 (planned live, not from the table): roll straight on while the
                # bar is still well ahead; else (and any later leg) wait.
                face = self.x2 - self.arena.obstacle_2_size[0] / 2.0 if self.x2 is not None else None
                if (not self.step and self.leg == 0 and face is not None
                        and abs(self.current_pose[2]) < math.radians(15.0)
                        and face - (self.current_pose[0] + self.car.footprint_front) > ROLL_ON_M):
                    self.hold_heading(float(self.p('path_speed')), 0.0)
                else:
                    self.send_cmd(0.0, 0.0)
                return
            if not path:
                self.send_cmd(0.0, 0.0)
                self.get_logger().error(f"LEG {self.leg + 1}/3   NO PATH - stopping")
                self.set_state(State.STOPPED)
                return
            if self.step and not self.step_go:     # step:=true: each leg waits for /step
                self.send_cmd(0.0, 0.0)
                note = f"leg {self.leg + 1}"
                if note != self._step_note:
                    self._step_note = note
                    self.get_logger().info(f"STEP      SPACE for leg {self.leg + 1}/3")
                return
            self.step_go = False
            self.set_path(self.trim_behind(path))
            gears = [pt[3] for pt in path]
            moves = ' '.join('fwd' if gears[i] >= 0 else 'REV'
                             for i in range(len(gears)) if i == 0 or gears[i] != gears[i - 1])
            self.get_logger().info(f"LEG {self.leg + 1}/3" + (f"   {moves}" if 'REV' in moves else ''))
            self.publish_waypoints()
        # The whole car inside the carpark -> done (no creeping on to its middle).
        if self.leg == len(self.waypoints) - 1 and self.in_carpark():
            return self.finish("all of it in the carpark")
        # Into the carpark: the ultrasonic sees the back wall -> done.
        r = self.fresh_range()
        if (self.leg == len(self.waypoints) - 1 and r is not None and r <= float(self.p('home_stop_dist'))
                and self.current_pose[0] < self.arena.opening_x):
            return self.finish("the back wall")
        self.publish_leg()
        self.check_stuck()
        if self.follower_step(stop_at_end=self.next_leg() is None):
            wx, wy, _ = self.waypoints[self.leg]
            off = math.hypot(self.current_pose[0] - wx, self.current_pose[1] - wy)
            name = f"CP{self.leg + 2}" if self.leg + 1 < len(self.waypoints) else 'HOME'
            self.get_logger().info(f"{name:<9} arrived {off * 100:.0f} cm off")
            self.leg += 1
            self.set_path([])
            if self.leg >= len(self.waypoints):
                self.finish("HOME")

    def trim_behind(self, path):
        """The path without its first points that the car has already passed: a leg
        planned from where the car was (rolling on meanwhile) - steering back to them
        would be a reverse or a full-lock loop."""
        x, y, yaw = self.current_pose
        c, s = math.cos(yaw), math.sin(yaw)
        k = 0
        while k < len(path) - 2 and path[k][3] > 0 and (path[k][0] - x) * c + (path[k][1] - y) * s < 0.0:
            k += 1
        return path[k:]

    def in_carpark(self) -> bool:
        """Every corner of the car's outline inside the carpark's walls (by IN_CARPARK_MARGIN_M)."""
        x, y, yaw = self.current_pose
        c, s, m = math.cos(yaw), math.sin(yaw), IN_CARPARK_MARGIN_M
        half = self.arena.carpark_width / 2.0 - m
        for a in (-self.car.footprint_rear, self.car.footprint_front):
            for b in (-self.car.footprint_half_width, self.car.footprint_half_width):
                px, py = x + c * a - s * b, y + s * a + c * b
                if not (TASK2_BACK_WALL_X + m < px < self.arena.opening_x - m and abs(py - TASK2_CENTRE_Y) < half):
                    return False
        return True

    def next_leg(self):
        """The next leg's path when the car rolls straight on into it: planned, and
        in the gear the current leg ends in. None otherwise (home, a gear change,
        still planning) - then the car slows for the checkpoint and stops there."""
        path = self.follower.path
        if self.step:                    # step:=true: stop at every checkpoint
            return None
        if self.state != State.FOLLOW or not path or self.leg + 1 >= len(self.leg_paths):
            return None
        nxt = self.leg_paths[self.leg + 1]
        return nxt if nxt and nxt[0][3] == path[-1][3] else None

    def follower_step(self, stop_at_end: bool = True) -> bool:
        """One tick along the follower's path at the speed profile. True once the
        path is done - then a stop command unless stop_at_end is False (the car
        rolls on while the next path is planned)."""
        speed = self.speed_ahead()
        look = self.car.lookahead_dist
        if self.car.use_velocity_scaled_lookahead_dist:
            look = max(look, look * speed / 0.2)
        self.follower.target_speed, self.follower.lookahead_dist = speed, look
        self.follower.tracking = str(self.p('path_tracking'))
        self.follower.pose_latency = float(self.p('pose_latency'))
        cmd = self.follower.compute_cmd()
        if cmd is None or self.follower.is_done():
            if stop_at_end:
                self.send_cmd(0.0, 0.0)
            return True
        self.send_cmd(*cmd)
        return False

    def speed_ahead(self) -> float:
        """Speed now: as fast as still lets the car brake, at brake_decel, down to every
        limit on the path ahead in time - each curve's sqrt(max_lateral_accel x R), and
        path_speed at a gear change or the path's end (not at checkpoint 1, nor at a
        checkpoint it rolls straight through). Between path_speed and straight_speed.
        The distances count from BRAKE_RESPONSE_S of travel ahead (the speed takes that
        long to follow the command). Replaced a flat slow_down_dist + lookahead window
        (0.85 m at 0.9 m/s), which slowed for every curve in it (sim 2026-10-12)."""
        path = self.follower.path
        v_min, v_max = float(self.p('path_speed')), float(self.p('straight_speed'))
        if not path:
            return v_min
        a_lat, a_brake = float(self.p('max_lateral_accel')), float(self.p('brake_decel'))
        x, y, _ = self.current_pose
        i0 = max(0, self.follower._search_idx - 40)
        k = min(range(i0, min(len(path), self.follower._search_idx + 1)),
                key=lambda j: (path[j][0] - x) ** 2 + (path[j][1] - y) ** 2)
        nxt = self.next_leg()
        path = path[k:] + (nxt[1:] if nxt else [])   # rolling on into the next leg: no slowing for the checkpoint
        margin = max(abs(self._last_cmd[0]), v_min) * BRAKE_RESPONSE_S
        horizon = margin + (v_max ** 2 - v_min ** 2) / (2.0 * a_brake) + 0.05

        def reachable(v_at, d):         # fastest speed now that still gets down to v_at in d
            return math.sqrt(v_at ** 2 + 2.0 * a_brake * max(0.0, d - margin))

        v, dist = v_max, 0.0
        for j in range(1, len(path)):
            ds = math.hypot(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1])
            dist += ds
            if path[j][3] != path[0][3]:
                return max(v_min, min(v, reachable(v_min, dist)))
            if ds > 1e-4:
                dth = math.atan2(math.sin(path[j][2] - path[j - 1][2]), math.cos(path[j][2] - path[j - 1][2]))
                kappa = abs(dth) / ds
                if kappa > 1e-6:
                    v = min(v, reachable(math.sqrt(a_lat / kappa), dist))
            if dist >= horizon:
                break
        else:
            # The end of the path is within reach. Checkpoint 1 is no stop: the car
            # rolls straight on and turns for obstacle 2 once it is planned.
            if self.state != State.TO_CHECKPOINT_1:
                v = min(v, reachable(v_min, dist))
        # cp1_speed after path_speed's floor: the straight into checkpoint 1 may be slower.
        return min(max(v_min, min(v_max, v, self.ir_pass_limit(a_brake, margin))), self.cp1_limit(a_brake, margin))

    def cp1_limit(self, a_brake: float, margin: float) -> float:
        """cp1_speed on the straight into checkpoint 1 (line_up_dist before it): steady
        and pointing straight while the ultrasonic and the camera read obstacle 2, arrow 2
        and the bar's width - at 0.9 m/s it arrived still turning (-13 deg) and wiggled
        (bags 2026-10-15)."""
        if self.state != State.TO_CHECKPOINT_1 or not self.lane_waypoint:
            return float('inf')
        v_slow = float(self.p('cp1_speed'))
        d = (self.lane_waypoint[0] - float(self.p('line_up_dist'))) - self.current_pose[0]
        if d <= 0.0:
            return v_slow
        return math.sqrt(v_slow ** 2 + 2.0 * a_brake * max(0.0, d - margin))

    def ir_pass_limit(self, a_brake: float, margin: float) -> float:
        """ir_pass_speed while an IR passes a face it is watching (and braking for one
        ahead): at full speed a 10 cm face got 2 readings at 25 Hz - too few for a fix
        (sim 2026-10-12, drift runs: most fixes refused)."""
        v_pass = float(self.p('ir_pass_speed'))
        x, y, yaw = self.current_pose
        c, s = math.cos(yaw), math.sin(yaw)
        if abs(c) < 0.8:                   # not running along the course
            return float('inf')
        sign = math.copysign(1.0, c)
        limit = float('inf')
        for name, fp in self.passes:
            sx, sy, _ = self.ir_sensors.get(name, (0.0, 0.0, 0.0))
            px, py = x + c * sx - s * sy, y + s * sx + c * sy
            if abs(py - fp.face_y) > 0.5:
                continue
            d = (fp.x0 - px) if sign > 0 else (px - fp.x1)      # to the face's near corner
            if d <= 0.0 and ((px <= fp.x1) if sign > 0 else (px >= fp.x0)):
                limit = min(limit, v_pass)                    # on the face
            elif d > 0.0:
                limit = min(limit, math.sqrt(v_pass ** 2 + 2.0 * a_brake * max(0.0, d - margin)))
        return limit

    def finish(self, why: str):
        self.send_cmd(0.0, 0.0)
        self.set_path([])
        self.set_state(State.FINISHED)
        self.end_run(done=True)            # FINISHED to the tablet
        self.get_logger().info(f"FINISHED  {self.run_time():.1f} s" + (f" ({why})" if why != 'all of it in the carpark'
                                                                           else ''))
        self.publish_waypoints()

    # -------------------------------------------------------------- planning ----

    def build_costmap(self):
        self.costmap = self.make_costmap(self.x1, self.x2)

    def make_costmap(self, x1, x2):
        """The course as far as it is known: carpark walls, the obstacles measured
        so far, and the side walls that MAY stand beside obstacle 2 (assumed there)."""
        return task2_routes.course_costmap(self.arena, x1, x2, self.bar_w(), float(self.p('obstacle_padding')),
                                           float(self.p('wall_padding')), float(self.p('obstacle_1_padding')))

    def plan_legs(self, start, gen, leg_paths, waypoints, costmap, ready=None):
        """Background thread: every leg back to back (like task 1). ready: legs
        already planned (the route table's, clean geometry)."""
        try:
            t0 = time.monotonic()
            pose = start
            for idx, target in enumerate(waypoints):
                # Clean geometry first (fast to drive); Hybrid A* only if it doesn't fit.
                path = (ready[idx] if ready and ready[idx] else None) or self.geometric_leg(idx, pose, target, costmap)
                if not path:
                    path = self.hybrid_leg(costmap, pose, target)
                if gen != self._plan_gen:
                    return
                leg_paths[idx] = path
                self.path_pub.publish(markers.route_path(leg_paths, self.stamp()))
                if not path:
                    self.get_logger().warn(f"PLAN      leg {idx + 1}/3 NO PATH")
                    return
                pose = target
            first = 'leg 1 table, legs 2-3' if self._leg1_note == 'table' else f"leg 1 {self._leg1_note}, legs 1-3"
            self.get_logger().info(f"PLAN      {first} in {time.monotonic() - t0:.1f} s")
        except Exception:
            self.get_logger().error(f"PLAN      crashed:\n{traceback.format_exc()}")

    # -------------------------------------------------------------- output ----

    def publish_map(self):
        stamp = self.stamp()
        back = TASK2_BACK_WALL_X
        cy, half = TASK2_CENTRE_Y, self.arena.carpark_width / 2.0
        self.arena_pub.publish(markers.arena_markers(
            stamp, size=TASK2_AREA_M, start_box=(back, cy - half, self.arena.opening_x, cy + half)))
        blocks, sizes, labels = [], [], []
        x1 = self.x1 if self.x1 is not None else self.x1_seen
        # The bar is drawn once measured - or at once when `pixi run setup` gave it.
        x2 = self.x2 if self.x2 is not None else (self.known()[1] if self.known() else None)
        bar = (self.arena.obstacle_2_size[0], self.bar_w())
        for i, (x, size) in enumerate(((x1, self.arena.obstacle_1_size), (x2, bar))):
            if x is not None:
                blocks.append((x, cy, 'W'))
                sizes.append(size[:2])
                labels.append(str(i + 1))
        walls = self.arena.carpark_walls() + (self.arena.side_walls(x2, self.bar_w()) if x2 is not None else [])
        self.obstacle_pub.publish(markers.obstacle_markers(blocks, labels, stamp, sizes=sizes, walls=walls))
        if self.costmap is not None:
            self.grid_pub.publish(markers.costmap_grid(self.costmap, stamp))

    def publish_waypoints(self):
        """The checkpoints: L beside obstacle 1, then A, B, HOME (numbered 1-4)."""
        shown = ([self.lane_waypoint] if self.lane_waypoint else []) + list(self.waypoints)
        if not shown:
            return
        current = None
        if self.state == State.TO_CHECKPOINT_1:
            current = 0
        elif self.state == State.FOLLOW:
            current = self.leg + (1 if self.lane_waypoint else 0)
        self.checkpoint_pub.publish(markers.checkpoint_markers(shown, current, self.stamp()))

    def publish_leg(self):
        path = self.leg_paths[self.leg] if self.leg < len(self.leg_paths) else None
        if path:
            self.leg_pub.publish(markers.leg_markers(path, self.follower.last_target, self.current_pose,
                                                     self.stamp()))

    def publish_search_progress(self, points_m):
        self.search_pub.publish(markers.search_progress(points_m, self.stamp()))

    def publish_run_status(self):
        """/run_status - the same message as task 1 (fields in RunStatus.msg)."""
        msg = RunStatus()
        msg.header.stamp, msg.header.frame_id = self.stamp(), 'map'
        msg.state = self.state.name
        msg.plan_state, msg.reset_done = self.plan_state(), bool(self.reset_done)
        msg.x, msg.y, msg.yaw = (float(v) for v in self.current_pose)
        msg.leg_count = 4
        msg.dist_to_checkpoint = float('nan')
        # The checkpoint being driven to: 1 beside obstacle 1, 2 / 3 beside obstacle
        # 2's ends, 4 = home. In `obstacle` (Foxglove "heading to checkpoint").
        target, label = None, ''
        if self.state in (State.TO_CHECKPOINT_1, State.APPROACH_1, State.SWERVE_OUT, State.SWERVE_BACK):
            target, label, msg.leg = self.lane_waypoint, '1', 1
        elif self.state == State.FOLLOW and self.leg < len(self.waypoints):
            target, msg.leg = self.waypoints[self.leg], self.leg + 2
            label = '4 (home)' if msg.leg == 4 else str(msg.leg)
        msg.obstacle = label
        if target is not None:
            wx, wy, wt = target
            msg.checkpoint_x, msg.checkpoint_y, msg.checkpoint_yaw = float(wx), float(wy), float(wt)
            msg.dist_to_checkpoint = math.hypot(self.current_pose[0] - wx, self.current_pose[1] - wy)
        if self.follower.active and self.follower.last_target is not None:
            tx, ty, _ = self.follower.last_target
            msg.wp_index, msg.wp_count = self.follower._search_idx + 1, len(self.follower.path)
            msg.target_x, msg.target_y = float(tx), float(ty)
        msg.cmd_v, msg.cmd_w = self._last_cmd
        msg.run_time = self.run_time()
        seen = self.arrow_counts[0] + self.arrow_counts[1]
        msg.yolo_ids, msg.yolo_counts = list(seen), list(seen.values())
        msg.detected_id = ' / '.join(a or '?' for a in self.arrows)
        self.run_status_pub.publish(msg)


def main(args=None):
    run(Task2Runner, args=args, events=True)   # ~330 msg/s of sensors: see run()


if __name__ == '__main__':
    main()
