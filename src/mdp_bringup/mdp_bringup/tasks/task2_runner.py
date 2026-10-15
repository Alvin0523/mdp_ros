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
                   straight bit the ultrasonic measures obstacle 2 (a bar on the
                   centre line) and the camera sees a bullseye on its near end
  CHECKPOINT_1     arrow 2 still unread: a glance (15 deg toward the centre line and
                   back, rolling - glance()). The bar's width from the gap between
                   arrow 2 and that bullseye in one frame x the ultrasonic's
                   distance (a heading error barely matters; else
                   bullseye + ultrasonic + pose; the usual 60 cm if none was seen),
                   to the nearest 5 cm ->
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
  FINISHED         in the carpark (or the ultrasonic sees the back wall); phase:=1
                   (launch argument): at checkpoint 1, stopped

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
US_VALID = (0.25, 2.2)        # m - lane readings outside this are not obstacle 2
# Beside obstacle 1 with arrow 2 or obstacle 2's distance still unknown: no stop -
# angle gently toward the centre line (the camera points at arrow 2, and the middle
# is as near to either end of the bar as anywhere) until both are known.
HEDGE_RAD = math.radians(7.0)
HEDGE_CENTRE_M = 0.10         # m off the centre line: straight on from there
# Arrow 2 still unread there: a glance - full lock GLANCE_RAD toward the centre line,
# held GLANCE_HOLD_S (a few camera frames), full lock straight again, rolling. From
# checkpoint 1 arrow 2 is ~32 deg off the camera's axis, its half view 31 deg (bag
# 2026-10-15, d2 = 0.6 m); 15 deg in, arrow 2 and the near bullseye are both in view.
# ~4 cm toward the centre line, within the route table's CP1 window (6 cm).
GLANCE_RAD = math.radians(15.0)
GLANCE_WAIT_S = 0.3           # s after it for YOLO's boxes of its frames (latency) - rolling on
ARROW_2_SURE = 3              # reads of arrow 2 before checkpoint 1 that make the glance unneeded
GLANCE_HOLD_S = 0.15
ARROW_2_DEADLINE_M = 0.60     # m from the bar's face with arrow 2 still unread: the nearer end
LEVEL_X_M = 0.30              # m before obstacle 1's centre: straight in the lane from here, the
                              # ultrasonic and camera see past it (obstacle 2 and arrow 2) and
                              # arrow 1 is out of the camera's view - read them on the move
BRAKE_RESPONSE_S = 0.15       # s for the wheels' speed to follow a slower command (speed_ahead)
IN_CARPARK_MARGIN_M = 0.02    # m: the whole outline this far inside the carpark's walls = home
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
        self.car = planner_params.ACTIVE          # URDF + navigation.yaml (robot.*, follower.*)
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

        # phase:=1 - carpark -> checkpoint 1 only: stop there and finish.
        self.phase = int(self.declare_parameter('phase', 0).value)

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
        self._glance = None              # None / 'turn' / 'hold' / 'back' / 'done' (glance())
        self._glance_since = None
        self.side = 0.0                  # +1 left / -1 right, the lane beside obstacle 1
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
            if not self.arrow_counts[i]:
                self.get_logger().info(f"ARROW {i + 1}   sees {side}")
            self.arrow_counts[i][side] += 1

    def measure_bar(self, x2: float):
        """The bar's width from the frames seen near checkpoint 1: (width or None,
        widths, 'gap' / 'pose') - arrow 2 to bullseye in one frame, else from the pose."""
        face_x = x2 - self.arena.obstacle_2_size[0] / 2.0
        return task2_vision.bar_width(self.bullseyes, self.arrows_2, float(self.p('camera_hfov')), face_x,
                                      TASK2_CENTRE_Y, self.side, self.arena.obstacle_2_width_range)

    def bar_w(self) -> float:
        """The bar's width planned with: measured (nearest 5 cm), set up, or the usual one."""
        if self.bar_width is not None:
            return self.bar_width
        if self.known() and self.known()[4] is not None:
            return self.known()[4]
        return self.arena.obstacle_2_size[1]

    def round_bar_width(self, width: float) -> float:
        """To the nearest 5 cm (the route table's step) - not up: the side walls stand a
        fixed distance from the bar's real ends (task2_routes.py, module docstring)."""
        return round(round(width / task2_routes.STEP_M) * task2_routes.STEP_M, 3)

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
        self.get_logger().info(f"E-STOP    at {self.fmt(self.current_pose)} after {self.run_time():.1f} s"
                               f" - reset before the next run")
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
            self.get_logger().info(f"START     arrow 1 {pre['arrow']}, obstacle 1 at "
                                   f"({markers.cell(self.x1)},{markers.cell(TASK2_CENTRE_Y)}) -> beside it "
                                   f"{self.fmt(pre['goal'])}")
            self.publish_map()
            self.set_state(State.TO_CHECKPOINT_1)
            self.publish_waypoints()
            return
        self.get_logger().info(f"START     from {self.fmt(self.current_pose)}, arrow 1 not read at home - "
                               f"driving on until it is")
        self.set_state(State.APPROACH_1)

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
        goal = (x1, TASK2_CENTRE_Y + SIDE[arrow] * float(self.p('lane_offset')), 0.0)
        self.pre = pre = {'arrow': arrow, 'x1': x1, 'goal': goal, 'path': None, 'planning': True}
        costmap = self.make_costmap(x1, None)

        # From the start pose, like task 1 - not the live pose: just after a reset that is
        # (0, 0) outside the course, and Hybrid A* searched max_planning_time (60 s) for
        # nothing (car 2026-10-14, with setup).
        def plan(start=self.start):
            t0 = time.monotonic()
            try:
                pre['path'] = (self.lined_up_path(start, goal, costmap, float(self.p('line_up_dist')))
                               or self.hybrid_leg(costmap, start, goal, (x1, None)))
                if not pre['path']:
                    self.get_logger().warn(f"PLAN      NO PATH start -> beside obstacle 1 from {self.fmt(start)} "
                                           f"after {time.monotonic() - t0:.1f} s - trying again")
                    return
                self.lane_waypoint = goal
                self.path_pub.publish(markers.route_path([pre['path']], self.stamp()))
                self.publish_waypoints()
                if not known:
                    self.get_logger().info(f"PLAN      start -> beside obstacle 1 {self.fmt(goal)} (arrow 1 {arrow})"
                                           f" - ready for START ({time.monotonic() - t0:.1f} s)")
                    return
                x2, arrow2 = known[1], known[3]
                waypoints = self.bar_waypoints(x2, arrow2)
                full = self.make_costmap(x1, x2)
                legs, pose = [], goal
                for idx, target in enumerate(waypoints):
                    path = self.geometric_leg(idx, pose, target, full) or self.hybrid_leg(full, pose, target, (x1, x2))
                    if not path:
                        self.get_logger().warn(f"PLAN      NO PATH to {self.names[idx]} {self.fmt(target)} - "
                                               f"trying again")
                        return
                    legs.append(path)
                    pose = target
                pre.update(x2=x2, arrow2=arrow2, waypoints=waypoints, legs=legs)
                self.waypoints = waypoints
                self.path_pub.publish(markers.route_path([pre['path']] + legs, self.stamp()))
                self.publish_waypoints()
                self.get_logger().info(f"PLAN      all 4 legs (setup layout, arrows {arrow} / {arrow2})"
                                       f" - ready for START ({time.monotonic() - t0:.1f} s)")
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

    def lined_up_path(self, start, goal, costmap, straight, margin=0.04):
        """start -> goal (same heading), straight again `straight` before it (task2_paths.py)."""
        return task2_paths.stepped_lined_up_path(start, goal, costmap, self.shapes(), straight, margin)

    def level_with_1(self, max_yaw: float = math.radians(8.0)) -> bool:
        """In the lane beside obstacle 1 and pointing (nearly) straight: the ultrasonic
        and the camera now look past it at obstacle 2."""
        return (self.x1 is not None and self.current_pose[0] >= self.x1 - LEVEL_X_M
                and abs(self.current_pose[2]) < max_yaw)

    def read_obstacle_2(self):
        """One ultrasonic reading of obstacle 2 (its centre along the course), kept
        if it is beyond obstacle 1."""
        if self.known():
            self.lane_readings = [self.known()[1]]
            return
        r = self.fresh_range(0.1)
        if r is not None and US_VALID[0] <= r <= US_VALID[1] and self.level_with_1(HEDGE_RAD + math.radians(4.0)):
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
        if not self.follower_step(stop_at_end=False):
            return
        # Keep rolling straight on while checkpoints 2-4 are planned (no stop).
        self.hold_heading(max(abs(self._last_cmd[0]), float(self.p('path_speed'))), 0.0)
        self.arrive_checkpoint_1()

    def arrive_checkpoint_1(self):
        if self.phase == 1:
            self.send_cmd(0.0, 0.0)
            self.set_path([])
            self.set_state(State.FINISHED)
            self.end_run(done=True)
            self.get_logger().info(f"FINISHED  phase 1 at checkpoint 1 {self.fmt(self.current_pose)} "
                                   f"in {self.run_time():.1f} s")
            return
        self.get_logger().info(f"CHECKPOINT 1 at {self.fmt(self.current_pose)}")
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
            goal = (x1, TASK2_CENTRE_Y + SIDE[arrow] * float(self.p('lane_offset')), 0.0)
            if self.costmap is None or self.x1_seen is None or abs(self.x1_seen - x1) > 0.03:
                self.x1_seen, self.costmap = x1, self.make_costmap(x1, None)
            x, y, _ = self.current_pose
            path = self.lined_up_path((x, y, 0.0), goal, self.costmap, float(self.p('line_up_dist')))
            if path:
                self.arrows[0], self.x1, self.side, self.lane_waypoint = arrow, x1, SIDE[arrow], goal
                self.cp1_planned = True
                self.set_path(path)
                self.watch_obstacle_1()
                self.get_logger().info(f"ARROW 1   {arrow} at {self.fmt(self.current_pose)}, obstacle 1 at "
                                       f"({markers.cell(x1)},{markers.cell(TASK2_CENTRE_Y)}) -> beside it "
                                       f"{self.fmt(goal)}")
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
        self.get_logger().info(f"SWERVE    {self.arrows[0]} at {self.fmt(self.current_pose)}, obstacle 1 at "
                               f"({markers.cell(self.x1)},{markers.cell(TASK2_CENTRE_Y)}), "
                               f"{math.degrees(self.swerve_angle):.0f}deg each way")
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
        """At checkpoint 1: obstacle 2's distance (ultrasonic, checked by the camera),
        arrow 2 (YOLO) and the bar's width (a bullseye) -> checkpoints 2, 3, 4 and the
        path through them, from right here. A reading still missing: no stop - roll
        on, angled gently toward the centre line (hedge()), until it comes; arrow 2
        still unread ARROW_2_DEADLINE_M from the bar: the nearer end."""
        if self.pre and self.pre.get('legs'):
            return self.follow_setup_legs()
        self.read_obstacle_2()
        if self.arrows[1] is None and (self.known() or sum(self.arrow_counts[1].values()) >= ARROW_2_SURE):
            self.arrows[1] = self.arrow(1)
        x, y, _ = self.current_pose
        x2_us = statistics.median(self.lane_readings) if self.lane_readings else None
        x2_cam = self.bar_x_from_camera()
        if x2_us is None and x2_cam is None:
            if x > self.x1 + NO_BAR_LIMIT_M:
                self.send_cmd(0.0, 0.0)
                self.get_logger().error("CHECKPOINT 1  obstacle 2 not seen by the ultrasonic or the camera - stopping")
                self.set_state(State.STOPPED)
                return
            return self.hedge()
        x2, why = self.choose_bar_x(x2_us, x2_cam)
        face = x2 - self.arena.obstacle_2_size[0] / 2.0
        near = face - (x + self.car.footprint_front) <= ARROW_2_DEADLINE_M
        if self.arrows[1] is None and not near:
            return self.hedge()
        # At d2 = 0.6 m the deadline is checkpoint 1 itself, and from there arrow 2 is
        # just outside the picture (sim 2026-10-14: 3 of 12 guessed): glance at it - also
        # when it is read but no frame had it with a bullseye yet (the width's gap
        # method; bag 2026-10-15: 2 glimpses of arrow 2 past obstacle 1 in the S, then
        # the pose method's 63 cm for 60). Once begun, it ends straight (leg 1 starts
        # facing +x).
        begun = self._glance not in (None, 'done')
        if begun or (near and self._glance is None and (self.arrows[1] is None or self.gap_missing(x2))):
            if self.glance():
                return
        # Still unread (YOLO's latency): stop and wait for it (then the nearer end) - a
        # guess is the wrong side half the time, a failed run.
        if self.arrows[1] is None and not self.need_arrow(1, 'LEFT' if self.side > 0 else 'RIGHT'):
            return
        if not self.decide_bar_width(x2):
            return self.hold_heading(max(abs(self._last_cmd[0]), float(self.p('path_speed'))), 0.0)
        if why:
            self.get_logger().warn(f"OBSTACLE 2 {why}")
        self.x2 = x2
        cy = TASK2_CENTRE_Y
        self.waypoints = self.bar_waypoints(self.x2, self.arrows[1])
        # Planned live (no table route): from where the car will be once the plan is
        # ready - it rolls on meanwhile (follow(), ROLL_ON_M); passed points are trimmed.
        x, y, _ = self.current_pose
        start = (x + abs(self._last_cmd[0]) * 0.15 + LIVE_PLAN_LEAD_M, y, 0.0)
        self.get_logger().info(f"OBSTACLE 2 at ({markers.cell(self.x2)},{markers.cell(cy)}), arrow 2 {self.arrows[1]}"
                               f", {self.bar_w() * 100:.0f} cm wide -> checkpoint 2 {self.fmt(self.waypoints[0])}"
                               f"  3 {self.fmt(self.waypoints[1])}  4 {self.fmt(self.home)}")
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
        self.get_logger().info(f"PLAN      {'leg 1 from the route table - ' if first else ''}"
                               f"{'legs 2, 3' if first else 'legs 1, 2, 3'} planning")
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

    def gap_missing(self, x2: float) -> bool:
        """The bar's width still to be measured, and no frame for the gap method yet."""
        if self.bar_width is not None or (self.known() and self.known()[4] is not None):
            return False
        width, _, how = self.measure_bar(x2)
        return width is None or how != 'gap'

    def decide_bar_width(self, x2: float) -> bool:
        """The bar's width for the rest of the run, from the bullseyes seen on the way
        into checkpoint 1. None seen yet while boxes still come: False - roll straight
        on up to bar_wait_dist past checkpoint 1 (YOLO's latency); then the usual width."""
        if self.bar_width is not None or (self.known() and self.known()[4] is not None):
            return True
        width, widths, how = self.measure_bar(x2)
        boxes_alive = self.boxes_seen_at is not None and self.now() - self.boxes_seen_at < 1.0
        if (how != 'gap' and boxes_alive and self._glance == 'done'
                and self.now() - self._glance_since < GLANCE_WAIT_S):
            self.get_logger().info("BAR       waiting for the glance's boxes - rolling on", throttle_duration_sec=1.0)
            return False
        if width is not None:
            self.bar_width = self.round_bar_width(width)
            each = ' '.join(f'{w * 100:.0f}' for w in widths[:8])
            self.get_logger().info(f"BAR       {width * 100:.1f} cm wide ({how}: {each}) -> planned as "
                                   f"{self.bar_width * 100:.0f} cm")
            return True
        cp1_x = self.lane_waypoint[0] if self.lane_waypoint else self.current_pose[0]
        if boxes_alive and self.current_pose[0] < cp1_x + float(self.p('bar_wait_dist')):
            self.get_logger().info("BAR       no bullseye yet - rolling on", throttle_duration_sec=1.0)
            return False
        self.bar_width = self.round_bar_width(self.arena.obstacle_2_size[1])
        self.get_logger().warn(f"BAR       no bullseye seen ({'no camera boxes' if not boxes_alive else 'none read'})"
                               f" - assuming the usual {self.bar_width * 100:.0f} cm")
        return True

    def table_route(self):
        """Checkpoint 1 -> 2 from the route table, in the map frame, or None (then
        planned live): only from the planned checkpoint 1 (not a swerve's end), with
        the car still near it and the table built for these settings."""
        if not (self.cp1_planned and self.lane_waypoint and self.routes_ok()):
            return None
        cp1 = self.lane_waypoint
        x, y, _ = self.current_pose
        if abs(y - cp1[1]) > 0.06 or not (CP1_WINDOW_M[0] < x - cp1[0] < CP1_WINDOW_M[1]):
            self.get_logger().warn(f"ROUTES    the car is off checkpoint 1 ({(x - cp1[0]) * 100:+.0f} / "
                                   f"{(y - cp1[1]) * 100:+.0f} cm) - planning leg 1 live")
            return None
        d2 = (self.x2 - self.arena.obstacle_2_size[0] / 2.0) - (self.x1 + self.arena.obstacle_1_size[0] / 2.0)
        local, why = self.routes.lookup(d2, self.bar_w(), self.arrows[0], self.arrows[1])
        if local is None:
            self.get_logger().warn(f"ROUTES    {why} - planning leg 1 live")
            return None
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
        self.get_logger().info(f"OBSTACLE 2 at ({markers.cell(x2)},{markers.cell(TASK2_CENTRE_Y)}), arrow 2 {arrow2}"
                               f" (setup) -> legs to checkpoints 2, 3, 4 planned before START")
        self.build_costmap()
        self.publish_map()
        self.watch_bar_ends()
        self.leg, self.plan_start = 0, self.pre['goal']
        self.set_state(State.FOLLOW)
        self.publish_waypoints()
        self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))

    def glance(self) -> bool:
        """Turn GLANCE_RAD toward the centre line and straight back, rolling at
        path_speed - arrow 2 and the near bullseye in one frame (the bar's width,
        gap method). True while it lasts; once only."""
        if self._glance == 'done':
            return False
        toward = -self.side                # +1 left
        yaw = self.current_pose[2] * toward
        v = float(self.p('path_speed'))
        if self._glance is None:
            self._glance = 'turn'
            self.get_logger().info(f"GLANCE    arrow 2 not read - turning {math.degrees(GLANCE_RAD):.0f} deg "
                                   f"{'left' if toward > 0 else 'right'} to see it")
        if self._glance == 'turn':
            if yaw < GLANCE_RAD - ARC_LEAD_RAD:
                self.steer(v, toward * math.pi)          # full lock (clamped)
                return True
            self._glance, self._glance_since = 'hold', self.now()
        if self._glance == 'hold':
            if self.now() - self._glance_since < GLANCE_HOLD_S:
                self.hold_heading(v, toward * GLANCE_RAD)
                return True
            self._glance = 'back'
        if self._glance == 'back':
            if yaw > ARC_LEAD_RAD:
                self.steer(v, -toward * math.pi)
                return True
            self._glance, self._glance_since = 'done', self.now()
            self.get_logger().info(f"GLANCE    straight again at {self.fmt(self.current_pose)}, arrow 2 "
                                   f"{self.arrow(1) or 'still not read'}")
        return False

    def hedge(self):
        """Roll on at path_speed, angled HEDGE_RAD toward the centre line (straight
        once within HEDGE_CENTRE_M of it)."""
        off = (self.current_pose[1] - TASK2_CENTRE_Y) * self.side
        heading = -self.side * HEDGE_RAD if off > HEDGE_CENTRE_M else 0.0
        self.get_logger().info("CHECKPOINT 1  waiting for obstacle 2 / arrow 2 - rolling on",
                               throttle_duration_sec=1.0)
        self.hold_heading(float(self.p('path_speed')), heading)

    def follow(self):
        """Drive the planned legs; until leg 1 is ready, keep straight to its start."""
        if not self.follower.active:
            path = self.leg_paths[self.leg] if self.leg < len(self.leg_paths) else []
            if path is None:                   # still planning
                # Leg 1 (planned live, not from the table): roll straight on while the
                # bar is still well ahead; else (and any later leg) wait.
                face = self.x2 - self.arena.obstacle_2_size[0] / 2.0 if self.x2 is not None else None
                if (self.leg == 0 and face is not None and abs(self.current_pose[2]) < math.radians(15.0)
                        and face - (self.current_pose[0] + self.car.footprint_front) > ROLL_ON_M):
                    self.hold_heading(float(self.p('path_speed')), 0.0)
                else:
                    self.send_cmd(0.0, 0.0)
                return
            if not path:
                self.send_cmd(0.0, 0.0)
                self.get_logger().error(f"LEG {self.leg + 1}/3   NO PATH to {self.names[self.leg]} - stopping")
                self.set_state(State.STOPPED)
                return
            self.set_path(self.trim_behind(path))
            gears = [pt[3] for pt in path]
            moves = ' '.join('fwd' if gears[i] >= 0 else 'REV'
                             for i in range(len(gears)) if i == 0 or gears[i] != gears[i - 1])
            self.get_logger().info(f"LEG {self.leg + 1}/3   -> {self.names[self.leg]} "
                                   f"{self.fmt(self.waypoints[self.leg])}  {moves}")
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
            self.get_logger().info(f"ARRIVED   {self.names[self.leg]} at {self.fmt(self.current_pose)}")
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
        return max(v_min, min(v_max, v, self.ir_pass_limit(a_brake, margin), self.arrow_2_limit(a_brake, margin)))

    def arrow_2_limit(self, a_brake: float, margin: float) -> float:
        """path_speed on the way into checkpoint 1 while arrow 2 is still unread: YOLO
        gets more frames while it is in view (it is near the picture's edge there, and
        out of view on the S before - sim 2026-10-14)."""
        if self.state != State.TO_CHECKPOINT_1 or self.x1 is None or self.arrow(1) is not None:
            return float('inf')
        v_slow = float(self.p('path_speed'))
        d = (self.x1 - 2 * LEVEL_X_M) - self.current_pose[0]          # to the straight bit
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
        self.get_logger().info(f"FINISHED  in the carpark at {self.fmt(self.current_pose)} ({why}) "
                               f"in {self.run_time():.1f} s")
        self.publish_waypoints()

    # -------------------------------------------------------------- planning ----

    def build_costmap(self):
        self.costmap = self.make_costmap(self.x1, self.x2)

    def make_costmap(self, x1, x2):
        """The course as far as it is known: carpark walls, the obstacles measured
        so far, and the side walls that MAY stand beside obstacle 2 (assumed there)."""
        return task2_routes.course_costmap(self.arena, x1, x2, self.bar_w(), float(self.p('obstacle_padding')),
                                           float(self.p('wall_padding')))

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
                    self.get_logger().warn(f"PLAN      leg {idx + 1}/3 to {self.names[idx]}: NO PATH")
                    return
                pose = target
            self.get_logger().info(f"PLAN      done, 3 legs in {time.monotonic() - t0:.1f} s")
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
