"""
Task 1 runner: visit every obstacle, read its image, report it to the tablet.

  WAITING_FOR_SETUP --/obstacle_setup--> PLANNING_PATH --> WAITING_FOR_START
      --/start_run--> NAVIGATING_TO_TARGET <--> PAUSE_FOR_SCAN --> FINISHED
  /stop_run -> STOPPED from anywhere; a pose reset (/reset_pose -> /set_pose) goes back to
  WAITING_FOR_START, keeping the obstacles and the plan.

Planning (mdp_algorithm): the visiting order and checkpoints are computed at
once; every leg's Hybrid A* path is then planned in a background thread, so
leg 2 is being planned while the car drives leg 1. The car drives each leg
with the pure pursuit follower, stops at the checkpoint for scan_pause_s while
YOLO looks, and sends TARGET,<obstacle>,<id> (the most frequent detection).

visit.mode (navigation.yaml): square = a square-on stop per image (above);
anchors = fewer stops (mdp_algorithm planning/visit_plan.py): square ANCHOR stops
(IR fix + scan), DRIVE-BY images read from the passing camera (/yolo_detections,
matched to the image by the box's direction) at visit.drive_by_speed, and an
angled LAST stop. A leg bent through a VIA point is planned as two Hybrid A*
legs joined into one. A drive-by not read on its leg is visited square at the
end. Either mode: what YOLO reads in the last APPROACH_READ_M before a stop
already counts toward its scan.

Settings: config/navigation.yaml (robot.* / costmap.* / planner.* / follower.*,
declared here and handed to the planner library).

Topics:
  in   /obstacle_setup  "id:x,y,facing|..." (metres), from the tablet bridge
       /yolo_result     detected symbol id
       /yolo_detections every box of a frame (drive-bys, reads on the approach)
       /odometry/filtered  EKF pose, converted to the `map` (arena) frame via TF
       /manual_drive    f/b/fl/fr/bl/br bursts from the tablet
       /set_pose        the EKF being reset (-> RESET confirmation)
       /ir, /ir2        the left IRs - the position fix at each scan stop (ir_pose_fix.py)
  out  /cmd_vel         the only /cmd_vel publisher; streams zeros when idle
       /bluetooth_tx    PLAN / RESET / STATUS / TARGET lines to the tablet
       /rosout          one log line per run event (launch terminal, Foxglove Log panel)
       /run_status      live numbers, 2 Hz (mdp_interfaces/RunStatus)
       Foxglove drawings - see mdp_bringup/utils/markers.py
  services  /start_run  /stop_run   (reset: /reset_pose, robot_pose_feedback)
            /step       step:=true only (`pixi run step`, SPACE)
            (/setup_obstacles - `pixi run setup`: the laptop's leg_planner)

step:=true (launch argument, testing): one obstacle per /step. The first starts the
run (like START); the car drives one leg and does everything it does on a real run
at that stop - the IR approach, creep and position fix, the scan, TARGET to the
tablet - then holds there until the next /step drives the next leg. The run timer
keeps running while it holds.
"""

import json
import math
import threading
import uuid
import traceback
from collections import Counter, deque
from enum import Enum, auto

import rclpy.time
from rclpy.parameter import Parameter
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from mdp_interfaces.msg import RunStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy
from robot_localization.srv import SetPose
from sensor_msgs.msg import Range
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from mdp_algorithm.control.path_follower import PathFollower, yaw_from_quaternion
from mdp_algorithm.planning.costmap import Costmap, Obstacle
from mdp_algorithm.planning.planner import plan_leg, plan_visiting_order
from mdp_algorithm.planning.visit_plan import plan_visits
from mdp_algorithm.planning.visiting_order import obstacle_to_checkpoint
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tasks.ir_pose_fix import (IR_APPROACH_M, IR_APPROACH_YAW, IR_CREEP_MAX_M, IR_FIX_DELAY_S,
                                           IR_FIX_MAX_S, IR_PAST_EDGE_M, IR_SEARCH_MAX_M, Face, IrPoseFix)
from mdp_bringup.tasks import task2_vision
from mdp_bringup.tasks.runner_base import RunnerBase
from mdp_bringup.utils.run import run
from mdp_bringup.utils import manual, markers, targets

# RESET turns DONE once odometry reports the start pose within these.
RESET_POS_TOL_M = 0.03
RESET_YAW_TOL_RAD = math.radians(5.0)
RESET_CONFIRM_TIMEOUT_S = 3.0
# A leg is planned again from where the car really is when it has no path, or
# the car is this far from where the leg starts (the previous leg failed or
# ended off its checkpoint). Legs are planned checkpoint to checkpoint ahead of
# time; in sim 2026-09-30 a NO PATH leg left the car following the next leg from
# a checkpoint it never reached, and it drove off the table.
REPLAN_POS_M = 0.05     # was 0.08 - kept under follower.max_path_error, or a leg would start
                        # already "off its path" and stop at once (2026-10-01)
OFF_PATH_RETRIES = 2    # plans per leg after leaving the path; then scan from where it stopped
REPLAN_YAW_RAD = math.radians(30.0)
REPLAN_FALLBACK_M = 0.15   # car inside the safety margin this close to the leg's start: plan from there
# After an IR fix the next leg is planned again (while the scan runs: no wait)
# if the car is this far off its start. 5 cm was not enough: 2-3 cm off, a leg
# starting in reverse left its path and replanned from the old start (sim 2026-10-03).
IR_REPLAN_POS_M = 0.02
SLID_IR_MAX_M = 0.03    # a stop slid further than this along the face: no IR creep or fix there
POSE_HISTORY_S = 2.0    # s of poses kept, to place a camera frame by its time stamp
APPROACH_READ_M = 0.35  # what YOLO reads for a stop's image this near it counts toward the scan
BOX_MATCH_RAD = math.radians(8.0)   # a box this near an image's expected direction is that image
VIEW_SLACK = 1.2        # the view limits x this when matching boxes (the plan's are tighter)


class State(Enum):
    WAITING_FOR_SETUP = auto()
    PLANNING_PATH = auto()
    WAITING_FOR_START = auto()       # planned; the car holds until /start_run
    NAVIGATING_TO_TARGET = auto()
    PAUSE_FOR_SCAN = auto()
    FINISHED = auto()
    STOPPED = auto()              # /stop_run: zeros until a pose reset


class Task1Runner(RunnerBase):
    def __init__(self):
        super().__init__('task1_runner')

        # Start pose in the map frame - the same numbers the launch file spawns
        # the car at and builds the map -> odom transform from.
        self.declare_parameter('start_x', 0.15)
        self.declare_parameter('start_y', 0.15)
        self.declare_parameter('start_yaw', math.pi / 2.0)
        # At a stop the car waits for YOLO: reads in the first scan_settle_s are
        # dropped, then it is done once the same id is read scan_confirm_count
        # times, or after scan_pause_s at most. Measured in sim 2026-09-30: the
        # third right read came 0.7-1.2 s after stopping, so a fixed 3 s wasted
        # ~2 s per obstacle. The settle: YOLO answers ~0.3 s after its frame, so
        # the first answers after stopping are of frames taken while still
        # turning in - a neighbour's image, or the image at an angle (read 36 as 33).
        self.declare_parameter('scan_pause_s', 3.0)
        self.declare_parameter('scan_settle_s', 0.5)
        self.declare_parameter('scan_confirm_count', 3)
        # The paths between checkpoints are planned by leg_planner on the laptop
        # (`pixi run laptop`); no answer within remote_plan_timeout s (or no new leg
        # for max_planning_time + that) and the car plans them itself.
        self.declare_parameter('remote_plan_timeout', 2.0)
        # At each scan stop: fix the pose against the block with the two left
        # IRs, creeping and centring on it first if needed (ir_stop_step).
        self.declare_parameter('ir_pose_fix', True)
        # The camera's horizontal view (rad): task2_runner's camera_hfov, passed by the launch.
        self.declare_parameter('camera_hfov', Parameter.Type.DOUBLE)
        # step:=true - one obstacle per /step (module docstring).
        self.step = bool(self.declare_parameter('step', False).value)
        self.create_service(Trigger, '/step', self.step_callback)
        self._step_hold = False        # step:=true: scanned, holding until the next /step
        manual.declare_params(self)   # the tablet's movement buttons, see utils/manual.py
        # robot.wheelbase / steering_limit_* (URDF) + navigation.yaml, passed
        # by the launch; the same files are the defaults for a bare `ros2 run`.
        self.planner_settings = list(planner_params.to_flat(planner_params.load()).items())
        for name, value in self.planner_settings:
            self.declare_parameter(name, value)
        self.configure_planner()

        # (/cmd_vel, /run_status, the Foxglove drawings, /start_run ...: RunnerBase)
        self.create_subscription(String, '/obstacle_setup', self.setup_callback, 10)
        self.create_subscription(String, '/yolo_result', self.yolo_callback, 10)
        self.create_subscription(String, '/yolo_detections', self.detections_callback, 10)
        self.poses = deque()             # (time s, pose): where the car was when a frame was taken
        # Manual drive lives here, not in the bridge: this node already owns /cmd_vel.
        self.create_subscription(String, '/manual_drive', self.manual_drive_callback, 10)
        plan_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.plan_request_pub = self.create_publisher(String, '/plan_legs/request', plan_qos)
        self.create_subscription(String, '/plan_legs/result', self.plan_result_callback, plan_qos)
        self._session = uuid.uuid4().hex[:8]   # tells this runner's requests from an earlier one's
        self._remote = None                    # the request the laptop is planning, see plan()
        self._obstacles_cm = []
        # Each scan stop's IR fix in the 3D view: an arrow old -> new pose + numbers.
        self.ir_fix_pub = self.create_publisher(MarkerArray, '/ir_fix_markers', 10)
        self._ir_fix_marks = []
        self._results = {}              # obstacle index -> (text, found): the 3D view's answers
        self._last_result = ''          # '#3 = Number 1 (11)': the run panel's scan result
        self._arrive_dir = 1.0          # +1 forward / -1 reverse: how the car drove into this stop
        self._laptop_ok = False                # the laptop answered this plan: replans go there too
        self._leg_waits = {}                   # gen -> a replan waiting for the laptop (remote_leg)
        self._replan_n = 0

        self.ir_fix = None             # IrPoseFix once the IR positions are known (plan)
        self._ir_fix_pending = False   # at a scan stop, the fix not yet applied
        self._finds = 0                # searches (start_find) at this stop - at most 2
        self._creep = None             # a straight creep in progress, see creep()
        for name in ('ir', 'ir2'):
            self.create_subscription(Range, f'/{name}', lambda m, n=name: self.ir_callback(n, m), 10)
        self.ekf_set_pose = self.create_client(SetPose, '/set_pose')   # the service: no /set_pose topic -> no RESET

        self.follower = PathFollower()
        self.current_pose = (0.0, 0.0, math.pi / 2)
        self.set_state(State.WAITING_FOR_SETUP)

        self.obstacles = []          # (x_m, y_m, facing) from /obstacle_setup
        self.tablet_ids = []         # the tablet's number for each obstacle
        self.visiting_order = []     # obstacle indices in visit order
        self.checkpoints = []        # (x, y, theta) per visit, same order
        self.leg_paths = []          # per visit: path, [] = no path, None = still planning
        self._replanned = set()      # legs already planned again this run (start_leg)
        self._heading_retried = set()  # legs re-driven once for arriving off heading (navigate)
        self._off_path = {}            # leg -> times it left its path (left_path)
        self._on_path = False          # car has reached its leg's path (left_path)
        self.unreachable = []
        self._slid = {}                # checkpoint index -> m slid along the face to fit (plan)
        # visit.mode anchors (visit_plan.py); square mode: every visit 'square', no vias/drive-bys.
        self.visit_kinds = []          # per visit: 'square' | 'anchor' | 'angled' | 'last'
        self.visit_vias = []           # per visit: poses its leg passes through (no stop)
        self.drive_bys = {}            # obstacle index -> visit index whose leg reads it
        self._drive_reads = {}         # obstacle index -> Counter of the ids read while driving
        self._drive_done = set()       # drive-by images already reported
        self._approach_reads = []      # ids read for the current stop's image on its approach
        self._planned_n = 0            # visits planned; more are added for missed drive-bys
        self._camera = None            # the camera on base_link (x, y, yaw), from the URDF
        # What Hybrid A* plans: every via and stop in order; a visit's leg is its parts joined.
        self.plan_targets = []         # poses
        self.target_visit = []         # per target: its visit index
        self.target_paths = []         # per target: path, [] = no path, None = planning
        self.target_starts = []        # per target: the pose it was planned from
        self.current_target_idx = 0
        self.costmap = None
        # Bumped whenever a plan is abandoned (new setup, stop): a planning thread
        # still running for the old one drops its results.
        self._plan_gen = 0

        # Tablet indicators (PLAN / RESET / STATUS lines).
        self.plan_state = 'WAITING'  # WAITING | PLANNING | DONE
        self.reset_done = False      # pose confirmed at the start
        self.stopped = False
        self._reset_pending = False
        self._reset_requested_at = 0.0
        self._last_sent = {}
        self._last_heartbeat = 0.0

        self._manual_cmd = (0.0, 0.0)
        self._manual_until = 0.0
        self._manual_active = False
        self.scan_detections = []
        self.detected_target_id = None
        self.get_logger().info("READY     waiting for obstacles")

    # ------------------------------------------------------------ helpers ----

    def start_pose(self):
        return tuple(float(self.get_parameter(n).value) for n in ('start_x', 'start_y', 'start_yaw'))

    def camera_yaw(self):
        """Which way the camera looks on the car (rad, +pi/2 = left): base_link ->
        camera_link from the URDF (mdp.launch.py sets camera_yaw per task). The
        checkpoint heading is turned by this so the CAMERA, not the nose, faces
        the image. None until robot_state_publisher's transform has arrived."""
        try:
            tf = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f'PLAN      waiting for base_link -> camera_link (URDF): {exc}',
                                   throttle_duration_sec=2.0)
            return None
        return yaw_from_quaternion(tf.transform.rotation)

    def camera_offset(self):
        """The camera on base_link (x, y, yaw), from the URDF; None until its transform arrives."""
        try:
            tf = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        t = tf.transform
        return (t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation))

    def ir_sensors(self):
        """Where the IRs are on the car (x, y, yaw on base_link), from the URDF;
        None while robot_state_publisher's transforms are missing."""
        sensors = {}
        for name in ('ir', 'ir2'):
            try:
                tf = self.tf_buffer.lookup_transform('base_link', f'{name}_link', rclpy.time.Time())
            except tf2_ros.TransformException:
                return None
            t = tf.transform
            sensors[name] = (t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation))
        return sensors

    def in_run(self) -> bool:
        return self.state in (State.NAVIGATING_TO_TARGET, State.PAUSE_FOR_SCAN)

    def tablet_id(self, obstacle_idx: int) -> str:
        return self.tablet_ids[obstacle_idx] if obstacle_idx < len(self.tablet_ids) else str(obstacle_idx + 1)

    def current_label(self) -> str:
        if self.current_target_idx < len(self.visiting_order):
            return self.tablet_id(self.visiting_order[self.current_target_idx])
        return '?'

    # ---------------------------------------------------------- callbacks ----

    def on_pose(self):
        """A pose reset turns DONE once the EKF reports the start pose."""
        now = self.now()
        self.poses.append((now, self.current_pose))
        while self.poses and now - self.poses[0][0] > POSE_HISTORY_S:
            self.poses.popleft()
        x, y, yaw = self.current_pose
        if self._reset_pending:
            sx, sy, syaw = self.start_pose()
            if (math.hypot(x - sx, y - sy) < RESET_POS_TOL_M
                    and abs(math.atan2(math.sin(yaw - syaw), math.cos(yaw - syaw))) < RESET_YAW_TOL_RAD):
                self._reset_pending = False
                self.reset_done = True
                self.get_logger().info("RESET     done - at the start")
            elif self.now() - self._reset_requested_at > RESET_CONFIRM_TIMEOUT_S:
                self._reset_pending = False
                self.get_logger().warn(
                    f"RESET     not confirmed after {RESET_CONFIRM_TIMEOUT_S:.0f} s - pose {self.fmt(self.current_pose)} "
                    f"is not the start {self.fmt(self.start_pose())} (EKF running? did it take /set_pose?)")

    def ir_callback(self, name, msg: Range):
        if self.ir_fix is not None and self.in_run():
            speed = abs(self.last_odom.twist.twist.linear.x) if self.last_odom is not None else 0.0
            self.ir_fix.on_reading(name, msg.range, self.current_pose, speed)

    def ir_stop_step(self, waited) -> bool:
        """A control tick at a scan stop with the IR fix due (ir_pose_fix.py):
        wait for readings; until both IRs see the face, creep the way it must be
        (start_find) and on IR_PAST_EDGE_M; fix the pose estimate. True while
        it holds the scan back."""
        if self._creep is not None:
            self.creep()
            return True
        if not self._ir_fix_pending:
            return False
        if waited < IR_FIX_DELAY_S or not (self.ir_fix.ready() or waited >= IR_FIX_MAX_S):
            return True
        # Not both on (stopped short, or rolled one off): search - at most twice.
        if self.ir_fix.seen() != 'both' and self._finds < 2:
            self._finds += 1
            if self.start_find():
                return True
        self._ir_fix_pending = False
        # Both IRs on the block (or as close as the search got): the car stays where
        # it is - the fix only puts the pose estimate where the car really is.
        fixed = self.apply_ir_fix()
        self.ir_fix.reset(None)
        self.check_next_leg(fixed)
        return False

    def ir_mid_x(self) -> float:
        """How far the middle of the two IRs is ahead of base_link (m, URDF)."""
        xs = [s[0] for s in self.ir_fix.sensors.values()]
        return sum(xs) / len(xs)

    def start_find(self) -> bool:
        """Not both IRs on the face: creep the way that brings the other on.
        Only the rear one sees it - the car is past the block, back; only the
        front one - short of it, forward; neither - back the way it came if an
        IR saw the block on the way in (drove past it), else on. True: creeping."""
        seen = self.ir_fix.seen()
        if seen in ('front', 'rear'):
            direction, dist = (1.0 if seen == 'front' else -1.0), IR_CREEP_MAX_M
            why = f"{seen} only"
        else:
            # Neither: what the IRs saw on the way in, not the pose (it is the pose
            # that is wrong). One saw the block, so the car drove past it - search
            # back the way it came; none did - it stopped short, search on.
            # (Sim 2026-10-07, car 7 cm off: asking the pose found nothing to do.)
            passed = self.ir_fix.seen_any
            direction, dist = (-self._arrive_dir if passed else self._arrive_dir), IR_SEARCH_MAX_M
            why = f"none, {'passed it' if passed else 'short of it'}"
        self._creep = {'mode': 'find', 'dir': direction, 'dist': dist, 'from': None, 'after': self.now(),
                       'seen': seen}
        self.get_logger().info(f"IR CREEP  #{self.current_label()} {why} - "
                               f"{'forward' if direction > 0 else 'back'} <={dist * 100:.0f} cm")
        return True

    def start_past(self, direction):
        """Both IRs on: creep on until IR_PAST_EDGE_M past where the second came onto
        the face (from that reading's pose, so the detection delay is not added)."""
        start = self.ir_fix.second_on_pose() or self.current_pose
        self._creep = {'mode': 'past', 'dir': direction, 'dist': IR_PAST_EDGE_M, 'from': start, 'after': 0.0}

    def creep(self):
        """Straight at follower.creep_linear_vel: 'find' until both
        IRs see the face, 'return' back to where it started; then the scan starts over."""
        c = self._creep
        if self.now() < c['after']:
            self.send_cmd(0.0, 0.0)
            return
        if c['from'] is None:
            c['from'] = self.current_pose
        done = math.hypot(self.current_pose[0] - c['from'][0], self.current_pose[1] - c['from'][1])
        if c['mode'] == 'past' and self.ir_fix.face is not None:
            # Along the face: coming in still turning, the car also moves towards it.
            done = abs(self.ir_fix.along_offset(self.current_pose)[0] - self.ir_fix.along_offset(c['from'])[0])
        seen = self.ir_fix.seen() if c['mode'] == 'find' else None
        found = seen == 'both'
        if found:                                 # on, without stopping, to IR_PAST_EDGE_M
            self.get_logger().info(f"IR CREEP  #{self.current_label()} both on after {done * 100:.1f} cm")
            self.start_past(c['dir'])
            return
        # The IR that saw the block lost it before the other found it: wrong way, or
        # the other cannot see it - go back to where it started (car, 2026-10-07: it
        # crept the full 10 cm off the block).
        lost = c['mode'] == 'find' and c['seen'] in ('front', 'rear') and seen == 'none'
        if not found and not lost and done < c['dist']:
            self.send_cmd(c['dir'] * float(self.get_parameter('follower.creep_linear_vel').value), 0.0)
            return
        self.send_cmd(0.0, 0.0)
        self._creep = None
        if lost:
            self.get_logger().warn(f"IR CREEP  #{self.current_label()} {c['seen']} lost it after "
                                   f"{done * 100:.1f} cm - back to start")
            self.ir_fix.edges = []                # along: not to be trusted from this pass
            self._creep = {'mode': 'return', 'dir': -c['dir'], 'dist': done, 'from': None, 'after': self.now() + 0.2}
            return
        if c['mode'] == 'find' and not found and c['seen'] == 'none':
            # A search that found nothing: the other way (through where it started,
            # as far again), then - nothing that way either - back to the start; it
            # must never leave the car further off than it stopped (sim 2026-10-07).
            if not c.get('second'):
                self.get_logger().warn(f"IR CREEP  #{self.current_label()} nothing in {done * 100:.0f} cm - "
                                       f"other way")
                self._creep = {'mode': 'find', 'dir': -c['dir'], 'dist': done + c['dist'], 'from': None,
                               'after': self.now() + 0.2, 'seen': 'none', 'second': done}
            else:
                self.get_logger().warn(f"IR CREEP  #{self.current_label()} nothing either way - back to the stop")
                self._creep = {'mode': 'return', 'dir': -c['dir'], 'dist': done - c['second'], 'from': None,
                               'after': self.now() + 0.2}
            return
        if c['mode'] == 'find':
            self.get_logger().info(f"IR CREEP  #{self.current_label()} not both after {done * 100:.1f} cm")
        elif c['mode'] == 'past':
            self.get_logger().info(f"IR STOP   #{self.current_label()} {done * 100:.1f} cm past the edge - "
                                   f"both IRs on the block")
        self.ir_fix.at_stop()                     # the gaps: from standing still again
        self.scan_detections, self.detected_target_id = [], None
        self.set_state(State.PAUSE_FOR_SCAN)      # the scan starts over

    def apply_ir_fix(self):
        """Shift the EKF pose by what the IRs say (heading kept); returns the
        pose the car is now at in the map."""
        dx, dy, note = self.ir_fix.correction()
        if math.hypot(dx, dy) < 0.005 or self.last_odom is None:
            self.get_logger().info(f"IR FIX    #{self.current_label()} none: {note}")
            return self.current_pose
        try:   # the shift is in the map frame; the EKF wants odom
            tf = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"IR FIX    no odom <- map transform ({exc}) - not applied")
            return self.current_pose
        a = yaw_from_quaternion(tf.transform.rotation)
        odom = self.last_odom.pose.pose
        req = SetPose.Request()
        req.pose.header.frame_id = self.last_odom.header.frame_id
        req.pose.header.stamp = self.last_odom.header.stamp   # the EKF's own time base
        p = req.pose.pose.pose
        p.position.x = odom.position.x + math.cos(a) * dx - math.sin(a) * dy
        p.position.y = odom.position.y + math.sin(a) * dx + math.cos(a) * dy
        p.orientation = odom.orientation
        req.pose.pose.covariance = list(self.last_odom.pose.covariance)
        self.ekf_set_pose.call_async(req)
        x, y, yaw = self.current_pose
        fixed = (x + dx, y + dy, yaw)
        self.show_ir_fix(self.current_pose, fixed, f"{dx:+.3f} {dy:+.3f}")   # map x, y shift, m
        self.get_logger().info(f"IR FIX    #{self.current_label()} {note} -> {self.fmt(fixed)}")
        return fixed

    def show_ir_fix(self, before, after, text):
        """This run's IR fixes in the 3D view (/ir_fix_markers): an arrow from where the
        pose was to where the IRs put it, and how far. None where nothing moved."""
        self._ir_fix_marks += markers.ir_fix_markers(self.stamp(), len(self._ir_fix_marks), before, after, text)
        self.ir_fix_pub.publish(MarkerArray(markers=self._ir_fix_marks))

    def check_next_leg(self, pose):
        """Where the scan leaves the car: off the next leg's start, that leg is
        planned again now, while YOLO is still looking."""
        nxt = self.current_target_idx + 1
        if (nxt < len(self.leg_paths) and nxt not in self._replanned and self.leg_paths[nxt]
                and self.off_leg_start(nxt, pose, IR_REPLAN_POS_M)):
            self._replanned.add(nxt)
            self.leg_paths[nxt] = None
            self.get_logger().info(f"REPLAN    leg {nxt + 1}/{len(self.visiting_order)} from {self.fmt(pose)} "
                                   f"(IR fix)")
            threading.Thread(target=self.replan_leg, daemon=True, args=(nxt, pose, self._plan_gen)).start()

    def setup_callback(self, msg: String):
        """`id:x,y,facing|...` (metres). Replaces the previous set and plan, but
        never interrupts a run."""
        if self.in_run():
            self.get_logger().warn("OBSTACLES ignored - a run is in progress, stop it first")
            return
        obstacles, ids = [], []
        for item in msg.data.strip().split('|'):
            if ':' not in item:
                continue
            try:
                obs_id, data = item.split(':')
                x, y, face = data.split(',')[:3]
                x, y, face = float(x), float(y), face.strip().upper()
            except ValueError:
                self.get_logger().warn(f"OBSTACLES skipping malformed entry {item!r}")
                continue
            if face not in markers.FACING:
                self.get_logger().warn(f"OBSTACLES skipping #{obs_id}: facing {face!r} is not N/E/S/W")
                continue
            obstacles.append((x, y, face))
            ids.append(obs_id.strip())
        if not obstacles:
            self.get_logger().warn("OBSTACLES none valid - ignored")
            return

        self._plan_gen += 1
        self.obstacles, self.tablet_ids = obstacles, ids
        self.visiting_order, self.checkpoints, self.unreachable, self.leg_paths = [], [], [], []
        self.current_target_idx = 0
        self.follower.set_path([])
        self.plan_state = 'PLANNING'
        self.set_state(State.PLANNING_PATH)
        cells = '  '.join(f"#{oid} ({markers.cell(x)},{markers.cell(y)}){face}"
                          for oid, (x, y, face) in zip(ids, obstacles))
        self.get_logger().info(f"OBSTACLES {cells}")

    def yolo_callback(self, msg: String):
        target_id = msg.data.strip()
        if self.state != State.PAUSE_FOR_SCAN or not target_id:
            return
        if self.now() - self.state_start < float(self.get_parameter('scan_settle_s').value):
            return   # a frame from before the car stopped
        self.scan_detections.append(target_id)
        self.detected_target_id = Counter(self.scan_detections).most_common(1)[0][0]

    def start_run_callback(self, request, response):
        """/start_run (tablet START, `pixi run start`, Foxglove START): only from WAITING_FOR_START with
        the plan done and the pose reset; otherwise says why not."""
        not_ready = {
            State.WAITING_FOR_SETUP: "Not ready: no obstacle setup received yet.",
            State.PLANNING_PATH: "Not ready: still planning the path.",
            State.STOPPED: "Ignored: stopped - reset the pose first (pixi run reset).",
            State.FINISHED: "Not ready: run finished - put the car back and reset (pixi run reset).",
        }
        if self.state == State.WAITING_FOR_START and self.plan_state != 'DONE':
            reason = "Not ready: still planning the path."
        elif self.state == State.WAITING_FOR_START and not self.reset_done:
            reason = "Not ready: reset the pose first (pixi run reset)."
        elif self.state != State.WAITING_FOR_START:
            reason = not_ready.get(self.state, f"Ignored: run already in progress ({self.state.name}).")
        else:
            reason = None
        response.success = reason is None
        if reason:
            response.message = reason
            self.get_logger().warn(f"`start` rejected: {reason}")
        else:
            self.set_state(State.NAVIGATING_TO_TARGET)
            self.current_target_idx = 0
            self._step_hold = False
            self._manual_until = 0.0
            self.forget_added_stops()
            self._drive_reads, self._drive_done, self._approach_reads = {}, set(), []
            self.reset_done = False     # the car is leaving the start
            self.run_start, self.run_end = self.now(), None
            self._ir_fix_marks = []                  # a new run: last run's IR fixes off the 3D view
            self._results, self._last_result = {}, ''
            self.publish_map()
            self.ir_fix_pub.publish(MarkerArray(markers=[markers.Marker(action=markers.Marker.DELETEALL)]))
            self._replanned = set()
            self._heading_retried = set()
            self._off_path = {}
            response.message = "Run started - navigating to targets."
            self.get_logger().info(f"START     {' -> '.join(self.tablet_id(i) for i in self.visiting_order)}")
            self.publish_checkpoints()
        self.sync_indicators()
        return response

    def step_callback(self, request, response):
        """/step (step:=true): before the run = START; holding at a stop after its
        scan = drive the next leg."""
        if not self.step:
            response.success, response.message = False, "Not in step mode - launch with step:=true."
            return response
        if self.state == State.WAITING_FOR_START:
            return self.start_run_callback(request, response)
        if self.state == State.NAVIGATING_TO_TARGET and self._step_hold:
            self._step_hold = False
            response.success, response.message = True, "Step."
            return response
        response.success = False
        response.message = (f"Ignored: {self.state.name} - still at it." if self.in_run() else
                            f"Ignored: {self.state.name}" + (" - reset to run again." if self.state in
                                                              (State.FINISHED, State.STOPPED) else "."))
        return response

    def forget_added_stops(self):
        """Stops added for missed drive-bys belong to that run only."""
        n = self._planned_n
        for lst in (self.visiting_order, self.checkpoints, self.visit_kinds, self.visit_vias,
                    self.leg_paths, self.leg_starts):
            del lst[n:]

    def stop_run_callback(self, request, response):
        """/stop_run (tablet STOP, `pixi run stop`): halt, hold zeros until reset."""
        self.stop()
        response.success, response.message = True, "Stopped."
        return response

    def stop(self):
        self._plan_gen += 1
        self.stopped = True
        self.reset_done = False
        self._reset_pending = False
        self._manual_until = 0.0
        self._ir_fix_pending = False
        self._creep = None
        self._step_hold = False
        self.follower.set_path([])
        if self.plan_state == 'PLANNING':
            self.plan_state = 'WAITING'   # a pose reset replans from the kept obstacles
        self.set_state(State.STOPPED)
        self.send_cmd(0.0, 0.0)
        self.end_run()
        self.get_logger().info(f"E-STOP    at {self.fmt(self.current_pose)} after {self.run_time():.1f} s"
                               f" - reset before the next run")
        self.publish_checkpoints()
        self.sync_indicators()

    def set_pose_callback(self, msg: PoseWithCovarianceStamped):
        """The EKF pose was reset (/reset_pose: `pixi run reset`, the tablet's
        RESET). RESET turns DONE once odometry reports the start pose
        (odom_callback); a finished or stopped run is readied again. A reset
        during a run stops it first."""
        if self.in_run():
            self.get_logger().warn("RESET     during a run - stopping it")
            self.stop()
        self.reset_done = False
        self._reset_pending = True
        self._reset_requested_at = self.now()
        self.stopped = False
        self._manual_until = 0.0
        self.follower.set_path([])
        if self.state in (State.FINISHED, State.STOPPED):
            self.current_target_idx = 0
            if self.plan_state == 'DONE':
                self.set_state(State.WAITING_FOR_START)
            elif self.obstacles:
                self._plan_gen += 1
                self.leg_paths = []
                self.plan_state = 'PLANNING'
                self.set_state(State.PLANNING_PATH)
            else:
                self.set_state(State.WAITING_FOR_SETUP)
        self.sync_indicators()

    def manual_drive_callback(self, msg: String):
        """A tablet movement button (utils/manual.py): a short burst on /cmd_vel,
        then zeros. Ignored during a run and while stopped."""
        key = msg.data.strip().lower()
        cmd = manual.burst(self, key, planner_params.ACTIVE.wheelbase)
        if cmd is None:
            self.get_logger().warn(f"MANUAL    unknown command {msg.data!r}")
            return
        if self.stopped or self.in_run():
            self.get_logger().warn(f"MANUAL    {key!r} ignored - a run is in progress or stopped")
            return
        v, w, seconds = cmd
        self._manual_cmd = (v, w)
        self._manual_until = self.now() + seconds
        self._manual_active = True
        self.reset_done = False          # moved by hand: no longer at a verified start
        self._reset_pending = False

    # -------------------------------------------------------- tablet lines ----

    def status_text(self) -> str:
        if self.stopped:
            return 'Stopped'
        if self.state == State.NAVIGATING_TO_TARGET:
            return f'Going to obstacle {self.current_label()}'
        if self.state == State.PAUSE_FOR_SCAN:
            return f'Scanning obstacle {self.current_label()}'
        if self.state == State.FINISHED:
            return 'Finished'
        if self.state == State.WAITING_FOR_START and self.plan_state == 'DONE' and self.reset_done:
            return 'Ready'
        return 'Waiting'

    def sync_indicators(self):
        """PLAN / RESET / STATUS lines, sent when they change (and every 2 s, see
        control_loop, for a bridge that came up later)."""
        lines = {'PLAN': f'PLAN:{self.plan_state}',
                 'RESET': f'RESET:{"DONE" if self.reset_done else "WAITING"}',
                 'STATUS': f'STATUS:{self.status_text()}'}
        for key, line in lines.items():
            if self._last_sent.get(key) != line:
                self._last_sent[key] = line
                self.send_bt(line)

    # -------------------------------------------------------- control loop ----

    def control_loop(self):
        now = self.now()
        self.check_remote_plan()
        if now - self._last_heartbeat >= 2.0:
            self._last_heartbeat = now
            self._last_sent.clear()
        self.sync_indicators()

        if self._manual_until > now:
            self.send_cmd(*self._manual_cmd)
            return
        if self._manual_active:
            self._manual_active = False
            self.send_cmd(0.0, 0.0)

        if self.state == State.PLANNING_PATH:
            self.plan()
        elif self.state == State.NAVIGATING_TO_TARGET:
            self.navigate()
        elif self.state == State.PAUSE_FOR_SCAN:
            self.send_cmd(0.0, 0.0)
            if self.ir_stop_step(now - self.state_start):
                return
            confirmed = (self.detected_target_id is not None
                         and Counter(self.scan_detections)[self.detected_target_id]
                         >= int(self.get_parameter('scan_confirm_count').value))
            if confirmed or now - self.state_start >= float(self.get_parameter('scan_pause_s').value):
                self.finish_scan()
        elif self.state == State.STOPPED:
            self.send_cmd(0.0, 0.0)
        elif self.state in (State.WAITING_FOR_START, State.FINISHED):
            self.send_cmd(0.0, 0.0)   # idle: hold the car still

    def configure_planner(self):
        """Hand the planner the current parameter values - at startup and before
        every plan, so `ros2 param set` + a new obstacle set (`pixi run setup`)
        tries e.g. another planner.checkpoint_standoff without a restart."""
        planner_params.configure(planner_params.from_flat(
            {name: self.get_parameter(name).value for name, _ in self.planner_settings}))

    def plan(self):
        """Order + checkpoints now; every leg in a background thread."""
        self.configure_planner()
        camera_yaw = self.camera_yaw()
        if camera_yaw is None:
            return   # tried again next tick
        obstacles_cm = [(x * 100.0, y * 100.0, face) for x, y, face in self.obstacles]
        self.costmap = Costmap([Obstacle(x, y, face, i) for i, (x, y, face) in enumerate(obstacles_cm)])
        self.publish_map()
        start = self.start_pose()
        self._camera = self.camera_offset()
        anchors = str(self.get_parameter('visit.mode').value) == 'anchors'
        if anchors and self._camera is None:
            self.get_logger().warn("PLAN      no camera_link transform yet - planning square stops")
        if anchors and self._camera is not None:
            visits, self.unreachable, self.costmap = plan_visits(obstacles_cm, start, self._camera)
            self.visiting_order = [v.obstacle for v in visits]
            self.checkpoints = [v.pose for v in visits]
            self.visit_kinds = [v.kind for v in visits]
            self.visit_vias = [list(v.vias) for v in visits]
            self.drive_bys = {o: k for k, v in enumerate(visits) for o in v.drive_bys}
        else:
            self.visiting_order, self.checkpoints, self.unreachable, self.costmap = plan_visiting_order(
                obstacles_cm, start, theta_offset=camera_yaw)
            self.visit_kinds = ['square'] * len(self.checkpoints)
            self.visit_vias = [[] for _ in self.checkpoints]
            self.drive_bys = {}
        # Stops the planner slid along the face for the car to fit (visiting_order):
        # index -> how far. Their IRs are not centred on the block; creeping would drive off.
        self._slid = {}
        for i, (x, y, t) in enumerate(self.checkpoints):
            bx, by, facing = self.obstacles[self.visiting_order[i]]
            nx, ny = markers.FACING[facing]
            along = (x - bx) * -ny + (y - by) * nx
            # A slide up to SLID_IR_MAX_M leaves both IRs on the 10 cm face (the IR creep
            # centres them again): still an IR stop. With 6 cm block padding two ordinary
            # stops slid 1 cm and lost their IR fix (car, 2026-10-09).
            if abs(along) > SLID_IR_MAX_M or self.visit_kinds[i] in ('angled', 'last'):   # angled: no IR fix
                self._slid[i] = abs(along)
        sensors = self.ir_sensors()
        self.ir_fix = IrPoseFix(sensors) if sensors else None
        if sensors is None:
            self.get_logger().warn("PLAN      no ir_link / ir2_link in the URDF - no IR position fix")
        else:
            # Each stop puts the MIDDLE of the two IRs level with the block centre, not
            # base_link: they are not centred on it (URDF), and lined up on base_link the
            # front one sat 0.9 cm from the block's edge and flickered off it (car,
            # 2026-10-07). Centred on their middle, both are as far inside as can be.
            # A stop the planner slid along the face stays where it is.
            mid = self.ir_mid_x()
            self.checkpoints = [(x, y, t) if i in self._slid else (x - mid * math.cos(t), y - mid * math.sin(t), t)
                                for i, (x, y, t) in enumerate(self.checkpoints)]
            self.get_logger().info(f"PLAN      stops moved {-mid * 100:+.1f} cm along: the IR pair's middle "
                                   f"level with each block")
        for i, d in self._slid.items():
            if self.visit_kinds[i] in ('angled', 'last'):
                continue
            self.get_logger().info(f"PLAN      #{self.tablet_id(self.visiting_order[i])} stop slid {d * 100:.0f} cm "
                                   f"along the face to fit - no IR fix there")
        self.publish_map()
        self.publish_checkpoints()
        if self.unreachable:
            self.get_logger().warn(
                f"PLAN      no scan checkpoint for {' '.join('#' + self.tablet_id(i) for i in self.unreachable)} - skipped")
        self.get_logger().info(f"PLAN      order {' -> '.join(self.tablet_id(i) for i in self.visiting_order)}")
        if anchors:
            names = {'anchor': 'anchor', 'angled': 'angled stop (no square spot fits)', 'last': 'last, angled'}
            for k, i in enumerate(self.visiting_order):
                reads = [f"#{self.tablet_id(o)}" for o, v in self.drive_bys.items() if v == k]
                self.get_logger().info(f"PLAN      #{self.tablet_id(i)} {names[self.visit_kinds[k]]}"
                                       + (f", drive-by {' '.join(reads)} on the way" if reads else '')
                                       + (' (bent through a via point)' if self.visit_vias[k] else ''))
        self._planned_n = len(self.visiting_order)
        self.leg_paths = [None] * len(self.visiting_order)
        self.leg_starts = [None] * len(self.visiting_order)   # pose each leg was planned from
        self.plan_targets, self.target_visit = [], []
        for k, cp in enumerate(self.checkpoints):
            for via in self.visit_vias[k]:
                self.plan_targets.append(via)
                self.target_visit.append(k)
            self.plan_targets.append(cp)
            self.target_visit.append(k)
        self.target_paths = [None] * len(self.plan_targets)
        self.target_starts = [None] * len(self.plan_targets)
        self._replanned = set()
        self._heading_retried = set()
        self._off_path = {}
        self.current_target_idx = 0
        self.set_state(State.WAITING_FOR_START)
        self._obstacles_cm = obstacles_cm
        self._laptop_ok = False
        self.request_remote_plan(obstacles_cm, start)

    def plan_locally(self, start):
        self._remote = None
        threading.Thread(target=self.plan_legs, daemon=True,
                         args=(start, self._plan_gen, list(self.plan_targets), self.costmap)).start()

    def remote_leg(self, idx, start):
        """One leg planned on the laptop (leg_planner): the path, [] for no path,
        or None when it does not answer - then the caller plans it here."""
        self._replan_n += 1
        gen = f'{self._session}:{self._plan_gen}:leg{idx}:{self._replan_n}'
        wait = {'ack': threading.Event(), 'done': threading.Event(), 'path': None}
        self._leg_waits[gen] = wait
        self.plan_request_pub.publish(String(data=json.dumps({
            'gen': gen, 'single': True, 'obstacles_cm': [list(o) for o in self._obstacles_cm],
            'start': list(start), 'checkpoints': [list(self.checkpoints[idx])],
            'params': {name: self.get_parameter(name).value for name, _ in self.planner_settings}})))
        timeout = float(self.get_parameter('remote_plan_timeout').value)
        try:
            if not wait['ack'].wait(timeout):
                return None
            if not wait['done'].wait(timeout + float(self.get_parameter('planner.max_planning_time').value)):
                return None
            return wait['path']
        finally:
            self._leg_waits.pop(gen, None)

    def request_remote_plan(self, obstacles_cm, start):
        """Ask leg_planner (laptop) for every leg; answers in plan_result_callback."""
        gen = f'{self._session}:{self._plan_gen}'
        self._remote = {'gen': gen, 'start': start, 'acked': False, 'last': self.now()}
        self.plan_request_pub.publish(String(data=json.dumps({
            'gen': gen, 'obstacles_cm': [list(o) for o in obstacles_cm], 'start': list(start),
            'checkpoints': [list(c) for c in self.plan_targets],
            'params': {name: self.get_parameter(name).value for name, _ in self.planner_settings}})))

    def plan_result_callback(self, msg: String):
        data = json.loads(msg.data)
        wait = self._leg_waits.get(data.get('gen'))
        if wait is not None:             # a mid-run replan (remote_leg)
            if data.get('ack'):
                wait['ack'].set()
            else:
                wait['path'] = [tuple(p) for p in data['path']]
                wait['done'].set()
            return
        r = self._remote
        if r is None or data.get('gen') != r['gen']:
            return                       # an old request's answer
        r['last'] = self.now()
        if data.get('ack'):
            if not r['acked']:
                r['acked'] = True
                self._laptop_ok = True
                self.get_logger().info("PLAN      legs planned on the laptop")
            return
        idx = data['idx']
        if idx >= len(self.target_paths):
            return
        self.store_target(idx, tuple(data['start']), [tuple(p) for p in data['path']])
        if all(p is not None for p in self.target_paths):
            self._remote = None
            self.plan_state = 'DONE'
            self.get_logger().info(f"PLAN      done, {len(self.checkpoints)} legs - waiting for START")

    def check_remote_plan(self):
        """control_loop: plan here when the laptop does not answer."""
        r = self._remote
        if r is None:
            return
        if r['gen'] != f'{self._session}:{self._plan_gen}':
            self._remote = None          # abandoned (new setup / stop)
            return
        timeout = float(self.get_parameter('remote_plan_timeout').value)
        if r['acked']:
            timeout += float(self.get_parameter('planner.max_planning_time').value)
        if self.now() - r['last'] > timeout:
            self.get_logger().warn("PLAN      no answer from the laptop planner - planning here")
            self.plan_locally(r['start'])

    def plan_legs(self, start, gen, targets, costmap):
        """Background thread: every leg back to back (vias as legs of their own,
        joined by store_target()). Writes only single list items (atomic under the
        GIL); control_loop polls them."""
        try:
            pose = start
            for idx, target in enumerate(targets):
                if gen != self._plan_gen:
                    return    # abandoned (new setup / stop)
                path = plan_leg(costmap, pose, target, progress_callback=self.publish_search_progress)
                if gen != self._plan_gen:
                    return
                self.store_target(idx, pose, path)
                pose = target
            if gen == self._plan_gen:
                self.plan_state = 'DONE'
                self.get_logger().info(f"PLAN      done, {len(self.checkpoints)} legs - waiting for START")
        except Exception:
            self.get_logger().error(f"PLAN      crashed:\n{traceback.format_exc()}")

    def store_target(self, idx, start, path):
        """A planned leg to plan_targets[idx]. Once every part of a visit's leg is in,
        they are joined into leg_paths[visit] (a via is driven through, not stopped at).
        A part with no path: the visit's leg is [] - start_leg() plans it again
        straight from where the car is."""
        self.target_starts[idx] = start
        self.target_paths[idx] = path
        if not path:
            self.get_logger().warn(f"PLAN      leg {idx + 1}/{len(self.plan_targets)}: NO PATH")
        v = self.target_visit[idx]
        parts = [k for k, owner in enumerate(self.target_visit) if owner == v]
        if all(self.target_paths[k] is not None for k in parts):
            paths = [self.target_paths[k] for k in parts]
            self.leg_starts[v] = self.target_starts[parts[0]]
            self.leg_paths[v] = [q for p in paths for q in p] if all(paths) else []
            for o in (o for o, k in self.drive_bys.items() if k == v):
                seen = self.view_length(self.leg_paths[v], o)
                log = self.get_logger().info if seen > 0.0 else self.get_logger().warn
                log(f"PLAN      drive-by #{self.tablet_id(o)} in view for {seen * 100:.0f} cm of leg {v + 1}"
                    + ('' if seen > 0.0 else ' - it gets a stop at the end if not read'))
        self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))

    def navigate(self):
        if self.current_target_idx >= len(self.leg_paths):
            self.send_cmd(0.0, 0.0)
            self.set_state(State.FINISHED)
            self.get_logger().info("FINISHED  nothing to visit")
            return
        if self._step_hold:
            self.send_cmd(0.0, 0.0)    # step:=true: scanned - waiting for /step
            return
        if not self.follower.active and not self.start_leg():
            self.send_cmd(0.0, 0.0)    # this leg is still being planned
            return
        if self.state != State.NAVIGATING_TO_TARGET:
            return                     # start_leg() skipped an empty leg

        self.publish_leg()
        # Speed, lookahead and the slow-downs are live parameters (the follower
        # lowers the speed where it is tight and grows the lookahead with it).
        f, p = self.follower, lambda name: self.get_parameter(f'follower.{name}').value
        f.target_speed = min(0.5, max(0.05, float(p('desired_linear_vel'))))
        f.lookahead_dist = float(p('lookahead_dist'))
        f.velocity_scaled_lookahead = bool(p('use_velocity_scaled_lookahead_dist'))
        f.regulate = bool(p('use_regulated_linear_velocity_scaling'))
        f.max_lateral_accel = float(p('max_lateral_accel'))
        f.min_regulated_speed = float(p('regulated_linear_scaling_min_speed'))
        f.max_reverse_speed = float(p('max_reverse_linear_vel'))
        f.approach_dist = float(p('approach_velocity_scaling_dist'))
        f.min_approach_speed = float(p('min_approach_linear_velocity'))
        f.tracking = str(p('path_tracking'))
        f.k_e, f.k_theta = float(p('feedback_k_e')), float(p('feedback_k_theta'))
        f.preview_time = float(p('feedforward_preview_time'))
        f.lqr_weights = tuple(float(p(n)) for n in ('lqr_q_lateral', 'lqr_q_heading', 'lqr_q_steer', 'lqr_r'))
        f.steer_tau = float(p('steering_time_constant'))
        f.pose_latency = float(p('pose_latency'))
        if self.drive_by_in_view():           # a steady view for YOLO (visit.drive_by_speed, live)
            f.target_speed = min(f.target_speed, float(self.get_parameter('visit.drive_by_speed').value))
        ir_stop = self.ir_approach()
        if ir_stop:
            self.follower.set_path([])            # both IRs on the block: this is the stop
        cmd = None if ir_stop else self.follower.compute_cmd()
        if cmd is None or self.follower.is_done():
            idx = self.current_target_idx
            # Only the IR that meets the block first is on it: creep on, no stop in between.
            creep_on = not ir_stop and self.ir_active(idx) and self.ir_fix.seen() == self.leading_ir()
            if not (creep_on or ir_stop):
                self.send_cmd(0.0, 0.0)
            cx, cy, ct = self.checkpoints[idx]
            yaw = self.current_pose[2]
            dyaw = math.degrees(math.atan2(math.sin(yaw - ct), math.cos(yaw - ct)))
            # Arrived pointing too far off to scan (the camera looks sideways at the
            # image): plan this leg once more from here - the planner finds the
            # manoeuvre to the checkpoint heading. Same limit as the planner's own
            # goal check, so any path it accepts also passes here (2026-10-01: the
            # real car arrived at #6 61 deg off and scanned the wrong way).
            limit = math.degrees(float(self.get_parameter('planner.goal_yaw_tolerance').value))
            if abs(dyaw) > limit and idx not in self._heading_retried:
                self._heading_retried.add(idx)
                self._replanned.add(idx)          # one try: if this fails too, scan from there
                self.leg_paths[idx] = None
                self.get_logger().warn(f"HEADING   #{self.current_label()} arrived {dyaw:+.0f}deg off "
                                       f"(limit {limit:.0f}) - planning the leg again from "
                                       f"{self.fmt(self.current_pose)}")
                threading.Thread(target=self.replan_leg, daemon=True,
                                 args=(idx, tuple(self.current_pose), self._plan_gen)).start()
                return
            # What YOLO already read on the approach (APPROACH_READ_M) counts toward the scan.
            self.scan_detections = list(self._approach_reads)
            self.detected_target_id = (Counter(self.scan_detections).most_common(1)[0][0]
                                       if self.scan_detections else None)
            self._approach_reads = []
            self.set_state(State.PAUSE_FOR_SCAN)
            if self.ir_active(idx):
                self.ir_fix.at_stop()
                self._ir_fix_pending, self._finds = True, int(ir_stop or creep_on)
            how = ('both IRs on the block - creeping on' if ir_stop else
                   f'{self.leading_ir()} IR on the block - creeping on' if creep_on else 'scanning')
            self.get_logger().info(f"ARRIVED   #{self.current_label()} at {self.fmt(self.current_pose)} "
                                   f"(target {self.fmt((cx, cy, ct))}, heading {dyaw:+.0f}deg) - {how}")
            self.stop_timer_at_last()
            if creep_on:
                self.start_find()
            elif ir_stop:
                self.start_past(self._arrive_dir)
        elif self.left_path():
            return
        else:
            self.send_cmd(*cmd)

    def stop_timer_at_last(self):
        """The car has stopped at the last obstacle: the run time stops here, not after
        YOLO's scan - someone times the run by hand and stops when it stops (2026-10-09).
        If the last stop does an IR creep, the car still moves: the time stops at FINISHED."""
        idx = self.current_target_idx
        if self.pending_drive_bys():
            return                 # a drive-by not read yet: it may still need a stop at the end
        if idx == len(self.visiting_order) - 1 and not self.ir_active(idx) and self.run_end is None:
            self.end_run(done=True)
            self.get_logger().info(f"TIMER     stopped at {self.run_time():.1f} s - "
                                   f"the car stopped at the last obstacle")

    def ir_active(self, idx) -> bool:
        """The IR approach, creep and fix at stop `idx`: not at the last one (nothing
        is driven after it, a better pose is no use) nor at one slid clear of the table's
        edge or a block (one IR at most reaches the block, on its very edge - a wrong fix)."""
        return (self.ir_fix is not None and bool(self.get_parameter('ir_pose_fix').value)
                and idx not in self._slid and idx != len(self.visiting_order) - 1
                and self.visit_kinds[idx] not in ('angled', 'last'))

    def leading_ir(self) -> str:
        """The IR that meets the block first driving into the stop: 'front' forward, 'rear' reversing."""
        return 'front' if self._arrive_dir > 0 else 'rear'

    def ir_approach(self) -> bool:
        """The leg's last stretch (IR_APPROACH_M from the stop, IR fix on): the IR
        that meets the block first is on it - creep speed; both on - stop there
        (True): the leg ends, start_past() takes it on. Only where ir_active()."""
        f = self.follower
        f.speed_cap = math.inf
        idx = self.current_target_idx
        if not self.ir_active(idx) or self.ir_fix.face is None or not f.path or not f.on_last_segment():
            return False
        self._arrive_dir = 1.0 if f.path[-1][3] >= 0 else -1.0
        cx, cy, ct = self.checkpoints[idx]
        x, y, yaw = self.current_pose
        if (math.hypot(x - cx, y - cy) > IR_APPROACH_M
                or abs(math.atan2(math.sin(yaw - ct), math.cos(yaw - ct))) > IR_APPROACH_YAW):
            return False
        seen = self.ir_fix.seen()
        if seen == 'both':
            return True
        if seen == self.leading_ir():
            f.speed_cap = float(self.get_parameter('follower.creep_linear_vel').value)
        return False

    def left_path(self) -> bool:
        """Further than follower.max_path_error off the leg's path: stop, and plan
        the leg again from here (up to OFF_PATH_RETRIES times, then scan from
        where it stopped). The planner only checked the path for blocks - off it,
        nothing did (2026-10-01: 14 cm off, it drove into two blocks)."""
        err = self.follower.path_error()
        limit = float(self.get_parameter('follower.max_path_error').value)
        # Armed once the car is on the path: a leg may start up to REPLAN_POS_M
        # off it, or from where it was meant to start (replan_leg's fallback,
        # up to REPLAN_FALLBACK_M) - the follower drives onto it first. Past
        # twice the limit it stops anyway.
        if err <= limit / 2.0:
            self._on_path = True
        if err <= (limit if self._on_path else 2.0 * limit):
            return False
        self.send_cmd(0.0, 0.0)
        idx = self.current_target_idx
        tries = self._off_path.get(idx, 0) + 1
        self._off_path[idx] = tries
        self._replanned.add(idx)
        self.follower.set_path([])
        if tries > OFF_PATH_RETRIES:
            self.leg_paths[idx] = []              # start_leg: NO PATH - scanning from here
            self.get_logger().warn(f"OFF PATH  #{self.current_label()} {err * 100:.0f} cm off again - "
                                   f"giving up on this leg")
            return True
        self.leg_paths[idx] = None
        self.get_logger().warn(f"OFF PATH  #{self.current_label()} {err * 100:.0f} cm off the path "
                               f"(limit {limit * 100:.0f}) - stopped, planning again from "
                               f"{self.fmt(self.current_pose)}")
        threading.Thread(target=self.replan_leg, daemon=True,
                         args=(idx, tuple(self.current_pose), self._plan_gen)).start()
        return True

    def start_leg(self) -> bool:
        """Load the current leg into the follower if it is planned. False while it
        is still planning; an empty leg (no path) is skipped straight to scanning."""
        idx = self.current_target_idx
        path = self.leg_paths[idx]
        if path is None:
            return False
        if idx not in self._replanned and (not path or self.off_leg_start(idx)):
            self._replanned.add(idx)
            self.leg_paths[idx] = None
            why = 'no path' if not path else f'leg starts at {self.fmt(path[0][:3])}'
            self.get_logger().warn(f"REPLAN    leg {idx + 1}/{len(self.visiting_order)} from "
                                   f"{self.fmt(self.current_pose)} ({why})")
            threading.Thread(target=self.replan_leg, daemon=True,
                             args=(idx, tuple(self.current_pose), self._plan_gen)).start()
            return False
        self.follower.set_path(path)
        self._on_path = False          # left_path() arms once the car is on it
        if not path:
            self.get_logger().warn(f"LEG {idx + 1}/{len(self.visiting_order)}   #{self.current_label()}: NO PATH - "
                                   f"scanning from here")
            self.scan_detections, self.detected_target_id = [], None
            self.set_state(State.PAUSE_FOR_SCAN)
            self.stop_timer_at_last()
            return True
        if self.ir_fix is not None:
            bx, by, facing = self.obstacles[self.visiting_order[idx]]
            self.ir_fix.reset(Face(bx, by, *markers.FACING[facing]))
        self._approach_reads = []
        gears = [p[3] for p in path]
        cuts = [i for i in range(1, len(gears)) if gears[i] != gears[i - 1]]
        moves = ' '.join('fwd' if gears[a] >= 0 else 'REV' for a in [0] + cuts)
        self.get_logger().info(f"LEG {idx + 1}/{len(self.visiting_order)}   -> #{self.current_label()} "
                               f"{self.fmt(self.checkpoints[idx])}  {moves}")
        return True

    def off_leg_start(self, idx, pose=None, pos_tol=REPLAN_POS_M) -> bool:
        """Is the car further than REPLAN_POS_M / REPLAN_YAW_RAD from where leg idx
        was planned from? That pose, not the path's first point: the planner's
        path starts one step (5 cm) ahead of it, so every leg looked 5 cm off and
        was replanned for nothing (2026-10-01, up to 19 s per leg)."""
        x, y, yaw = pose or self.current_pose
        px, py, ptheta = self.leg_starts[idx][:3]
        dyaw = abs(math.atan2(math.sin(yaw - ptheta), math.cos(yaw - ptheta)))
        return math.hypot(x - px, y - py) > pos_tol or dyaw > REPLAN_YAW_RAD

    def replan_leg(self, idx, start, gen):
        """Background thread: one leg from the car's real pose (current settings)."""
        try:
            self.configure_planner()
            if self.costmap.in_collision(start[0] * 100.0, start[1] * 100.0, start[2]):
                # Stopped a little inside the safety margin (2 cm closer to the block
                # than planned is enough) - the planner refuses such a start. Plan
                # from where the leg was meant to start; the follower gets onto it.
                planned = self.checkpoints[idx - 1] if idx > 0 else self.start_pose()
                if math.hypot(start[0] - planned[0], start[1] - planned[1]) < REPLAN_FALLBACK_M:
                    self.get_logger().warn(f"REPLAN    leg {idx + 1}: the car is inside the safety margin - "
                                           f"planning from {self.fmt(planned)} instead")
                    start = planned
            path = None
            if self._laptop_ok:
                t0 = self.now()
                path = self.remote_leg(idx, start)
                if path is None:
                    self.get_logger().warn(f"REPLAN    leg {idx + 1}: laptop no answer - planning here")
                else:
                    self.get_logger().info(f"REPLAN    leg {idx + 1}: laptop {self.now() - t0:.1f} s")
            if path is None:
                path = plan_leg(self.costmap, start, self.checkpoints[idx],
                                progress_callback=self.publish_search_progress)
            if gen != self._plan_gen:
                return
            self.leg_starts[idx] = start
            self.leg_paths[idx] = path
            self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))
        except Exception:
            self.get_logger().error(f"REPLAN    crashed:\n{traceback.format_exc()}")
            self.leg_paths[idx] = []

    def report_target(self, obstacle, reads, how):
        """TARGET,<obstacle>,<id> to the tablet: the most frequent of `reads` (UNKNOWN if none)."""
        found = bool(reads)
        target_id = Counter(reads).most_common(1)[0][0] if found else "UNKNOWN"
        obs = self.tablet_id(obstacle)
        self.send_bt(f"TARGET,{obs},{target_id}")
        seen = ' '.join(f"{k}x{n}" for k, n in Counter(reads).most_common()) or 'nothing'
        self.get_logger().info(f"TARGET    #{obs} = {targets.label(target_id)}   (YOLO saw {seen}, {how})")
        self._last_result = f'#{obs} = {targets.label(target_id) if found else "UNKNOWN"}'
        self._results[obstacle] = (targets.short(target_id) if found else '?', found)
        self.publish_map()

    def finish_scan(self):
        """Report the most frequent detection, then go on to the next obstacle."""
        self.report_target(self.visiting_order[self.current_target_idx], self.scan_detections,
                           f"{self.now() - self.state_start:.1f} s")
        self.current_target_idx += 1
        if self.current_target_idx >= len(self.visiting_order) and not self.visit_missed_drive_bys():
            scan_s = self.now() - self.state_start
            self.set_state(State.FINISHED)
            self.end_run(done=True)        # FINISHED to the tablet here unless the timer stopped already
            self.get_logger().info(f"FINISHED  all obstacles visited in {self.run_time():.1f} s "
                                   f"(+{scan_s:.1f} s last scan)")
        else:
            self.set_state(State.NAVIGATING_TO_TARGET)   # the next tick loads the leg
            if self.step:
                self._step_hold = True
                self.get_logger().info(f"STEP      SPACE for leg {self.current_target_idx + 1}/"
                                       f"{len(self.visiting_order)} -> #{self.current_label()}")
        self.publish_checkpoints()

    # ----------------------------------------------------------- drive-bys ----

    def pose_at(self, t: float):
        """The car's pose at time t (the nearest kept), or the current one."""
        if not self.poses or t < self.poses[0][0] - 0.1:
            return self.current_pose
        return min(self.poses, key=lambda tp: abs(tp[0] - t))[1]

    def image_view(self, pose, obstacle, slack=1.0):
        """(distance m, bearing rad, + left) of the obstacle's image from the camera at
        `pose`, if it is in view (visit.view_* x slack) - else None."""
        if self._camera is None:
            return None
        cx_, cy_, cyaw = self._camera
        x, y, th = pose
        cx, cy = x + cx_ * math.cos(th) - cy_ * math.sin(th), y + cx_ * math.sin(th) + cy_ * math.cos(th)
        bx, by, facing = self.obstacles[obstacle]
        nx, ny = markers.FACING[facing]
        dx, dy = bx + 0.05 * nx - cx, by + 0.05 * ny - cy
        d = math.hypot(dx, dy)
        g = lambda name: float(self.get_parameter(f'visit.{name}').value)
        if d < 0.1 or d > g('view_max_dist') * slack:
            return None
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - th - cyaw), math.cos(math.atan2(dy, dx) - th - cyaw))
        if abs(bearing) > g('view_max_bearing') * slack:
            return None
        if math.acos(max(-1.0, min(1.0, -(dx * nx + dy * ny) / d))) > g('view_max_incidence') * slack:
            return None
        return d, bearing

    def view_length(self, path, obstacle) -> float:
        """The longest stretch (m) of `path` with the obstacle's image in view."""
        best = run = 0.0
        last = None
        for q in path:
            if self.image_view(q[:3], obstacle):
                if last is not None:
                    run += math.hypot(q[0] - last[0], q[1] - last[1])
                last = q
                best = max(best, run)
            else:
                run, last = 0.0, None
        return best

    def pending_drive_bys(self):
        return [o for o in self.drive_bys if o not in self._drive_done]

    def drive_by_in_view(self) -> bool:
        return self.state == State.NAVIGATING_TO_TARGET and any(
            self.image_view(self.current_pose, o) for o in self.pending_drive_bys())

    def detections_callback(self, msg: String):
        """/yolo_detections: every box of a frame. Each box goes to the image the camera
        expects in that direction (BOX_MATCH_RAD): a drive-by not read yet, or - in the
        last APPROACH_READ_M before a stop - that stop's image."""
        if self.state != State.NAVIGATING_TO_TARGET:
            return
        hfov = self.get_parameter('camera_hfov').value
        try:
            stamp, w, _, boxes = task2_vision.parse(msg.data)
        except (ValueError, KeyError, TypeError):
            return
        if hfov is None or not boxes:
            return
        pose = self.pose_at(stamp)
        idx = self.current_target_idx
        expect = {o: self.image_view(pose, o, VIEW_SLACK) for o in self.pending_drive_bys()}
        target = self.visiting_order[idx] if idx < len(self.visiting_order) else None
        if target is not None:
            cx, cy, _ = self.checkpoints[idx]
            if math.hypot(pose[0] - cx, pose[1] - cy) < APPROACH_READ_M:
                expect[target] = self.image_view(pose, target, VIEW_SLACK)
        expect = {o: v for o, v in expect.items() if v}
        if not expect:
            return
        for box in boxes:
            if not box[0].isdigit():
                continue                                   # not an image ID (e.g. a bullseye)
            b = task2_vision.box_bearing(box, w, float(hfov))
            o, v = min(expect.items(), key=lambda ov: abs(ov[1][1] - b))
            if abs(v[1] - b) > BOX_MATCH_RAD:
                continue
            if o == target and o not in self.drive_bys:
                self._approach_reads.append(box[0])
                continue
            reads = self._drive_reads.setdefault(o, Counter())
            reads[box[0]] += 1
            if reads.most_common(1)[0][1] >= int(self.get_parameter('scan_confirm_count').value):
                self._drive_done.add(o)
                self.report_target(o, list(reads.elements()), 'driving by')

    def visit_missed_drive_bys(self) -> bool:
        """End of the planned stops: a drive-by read but not confirmed is reported as
        read; one never seen gets a square stop now (planned from here). True if stops
        were added."""
        missed = []
        for o in self.pending_drive_bys():
            self._drive_done.add(o)
            if self._drive_reads.get(o):
                self.report_target(o, list(self._drive_reads[o].elements()), 'driving by, not confirmed')
            else:
                missed.append(o)
        added = []
        for o in missed:
            cp = obstacle_to_checkpoint(self.costmap, self.costmap.obstacles[o], self._camera[2])
            if cp is None:
                self.report_target(o, [], 'no stop for it')
                continue
            x, y, t = cp[0] / 100.0, cp[1] / 100.0, cp[2]
            if self.ir_fix is not None:
                mid = self.ir_mid_x()
                x, y = x - mid * math.cos(t), y - mid * math.sin(t)
            self.visiting_order.append(o)
            self.checkpoints.append((x, y, t))
            self.visit_kinds.append('anchor')
            self.visit_vias.append([])
            self.leg_paths.append(None)
            self.leg_starts.append(None)
            added.append(len(self.visiting_order) - 1)
            self.get_logger().warn(f"DRIVE-BY  #{self.tablet_id(o)} not read - a stop for it now")
        if not added:
            return False
        threading.Thread(target=self.plan_added, daemon=True, args=(added, tuple(self.current_pose), self._plan_gen)).start()
        self.set_state(State.NAVIGATING_TO_TARGET)    # each leg loads once it is planned
        self.publish_checkpoints()
        return True

    def plan_added(self, idxs, start, gen):
        """Background thread: the legs to the stops added for missed drive-bys."""
        for k in idxs:
            if gen != self._plan_gen:
                return
            self.replan_leg(k, start, gen)
            start = self.checkpoints[k]

    # -------------------------------------------------------------- output ----

    def publish_map(self):
        stamp = self.stamp()
        self.grid_pub.publish(markers.costmap_grid(self.costmap, stamp))
        self.arena_pub.publish(markers.arena_markers(stamp))
        self.obstacle_pub.publish(markers.obstacle_markers(
            self.obstacles, [self.tablet_id(i) for i in range(len(self.obstacles))], stamp,
            results=self._results))

    def publish_checkpoints(self):
        if self.checkpoints:
            current = self.current_target_idx if self.in_run() else None
            self.checkpoint_pub.publish(markers.checkpoint_markers(self.checkpoints, current, self.stamp()))

    def publish_leg(self):
        stamp = self.stamp()
        self.path_pub.publish(markers.route_path(self.leg_paths, stamp))
        path = self.leg_paths[self.current_target_idx]
        if path:
            self.leg_pub.publish(markers.leg_markers(path, self.follower.last_target, self.current_pose, stamp))

    def publish_search_progress(self, points_m):
        self.search_pub.publish(markers.search_progress(points_m, self.stamp()))

    def publish_run_status(self):
        """/run_status - fields in mdp_interfaces/msg/RunStatus.msg."""
        msg = RunStatus()
        msg.header.stamp, msg.header.frame_id = self.stamp(), 'map'
        msg.state, msg.plan_state, msg.reset_done = self.state.name, self.plan_state, bool(self.reset_done)
        msg.x, msg.y, msg.yaw = (float(v) for v in self.current_pose)
        msg.leg_count = len(self.visiting_order)
        msg.dist_to_checkpoint = float('nan')
        idx = self.current_target_idx
        if self.in_run() and idx < len(self.checkpoints):
            msg.obstacle, msg.leg = self.current_label(), idx + 1
            cx, cy, ct = self.checkpoints[idx]
            msg.checkpoint_x, msg.checkpoint_y, msg.checkpoint_yaw = float(cx), float(cy), float(ct)
            msg.dist_to_checkpoint = math.hypot(self.current_pose[0] - cx, self.current_pose[1] - cy)
        if self.state == State.NAVIGATING_TO_TARGET and self.follower.last_target is not None:
            tx, ty, gear = self.follower.last_target
            msg.wp_index, msg.wp_count = self.follower._search_idx + 1, len(self.follower.path)
            msg.target_x, msg.target_y, msg.reverse = float(tx), float(ty), gear < 0
        msg.cmd_v, msg.cmd_w = self._last_cmd
        msg.run_time = self.run_time()
        msg.scan_duration = float(self.get_parameter('scan_pause_s').value)
        if self.state == State.PAUSE_FOR_SCAN:
            msg.scan_elapsed = float(self.now() - self.state_start)
            counts = Counter(self.scan_detections).most_common()
            msg.yolo_ids, msg.yolo_counts = [k for k, _ in counts], [n for _, n in counts]
            msg.detected_id = self.detected_target_id or ''
        if self.in_run() and idx < len(self.visiting_order):
            msg.target_label = f'#{self.current_label()}'
        # Live while scanning, then the last answer until the next scan has one.
        if self.state == State.PAUSE_FOR_SCAN and self.detected_target_id:
            msg.scan_result = f'#{self.current_label()} = {targets.label(self.detected_target_id)}'
        else:
            msg.scan_result = self._last_result
        self.run_status_pub.publish(msg)


def main(args=None):
    run(Task1Runner, args=args)


if __name__ == '__main__':
    main()
