"""
Task 1 runner: visit every obstacle, read its image, report it to the tablet.

  WAITING_FOR_SETUP --/obstacle_setup--> PLANNING_PATH --> WAITING_FOR_GO
      --/start_run--> NAVIGATING_TO_TARGET <--> PAUSE_FOR_SCAN --> FINISHED
  /stop_run -> STOPPED from anywhere; a pose reset (/reset_pose -> /set_pose) goes back to
  WAITING_FOR_GO, keeping the obstacles and the plan.

Planning (mdp_algorithm): the visiting order and checkpoints are computed at
once; every leg's Hybrid A* path is then planned in a background thread, so
leg 2 is being planned while the car drives leg 1. The car drives each leg
with the pure pursuit follower, stops at the checkpoint for scan_pause_s while
YOLO looks, and sends TARGET,<obstacle>,<id> (the most frequent detection).

Settings: config/navigation.yaml (robot.* / costmap.* / planner.* / follower.*,
declared here and handed to the planner library).

Topics:
  in   /obstacle_setup  "id:x,y,facing|..." (metres), from the tablet bridge
       /yolo_result     detected symbol id
       /odometry/filtered  EKF pose, converted to the `map` (arena) frame via TF
       /manual_drive    f/b/fl/fr/bl/br bursts from the tablet
       /set_pose        the EKF being reset (-> RESET confirmation)
  out  /cmd_vel         the only /cmd_vel publisher; streams zeros when idle
       /bluetooth_tx    PLAN / RESET / STATUS / TARGET lines to the tablet
       /rosout          one log line per run event (`pixi run runlog`)
       /run_status      live numbers, 2 Hz (mdp_interfaces/RunStatus)
       Foxglove drawings - see mdp_bringup/utils/markers.py
  services  /start_run  /stop_run   (reset: /reset_pose, robot_pose_feedback)
"""

import json
import math
import threading
import uuid
import traceback
from collections import Counter
from enum import Enum, auto

import rclpy.time
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from mdp_interfaces.msg import RunStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from mdp_algorithm.control.pure_pursuit_follower import PurePursuitController, yaw_from_quaternion
from mdp_algorithm.planning.costmap import Costmap, Obstacle
from mdp_algorithm.planning.planner import plan_leg, plan_visiting_order
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tasks.runner_base import RunnerBase
from mdp_bringup.utils.run import run
from mdp_bringup.utils import manual, markers

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


class State(Enum):
    WAITING_FOR_SETUP = auto()
    PLANNING_PATH = auto()
    WAITING_FOR_GO = auto()       # planned; the car holds until /start_run
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
        # role:=car (a laptop runs `pixi run base`): the paths between checkpoints
        # are planned by task1_planner on the laptop; no answer within
        # remote_plan_timeout s (or no new leg for max_planning_time + that) and
        # the car plans them itself, as when it runs alone.
        self.declare_parameter('remote_planner', False)
        self.declare_parameter('remote_plan_timeout', 2.0)
        # true: don't stream zeros while idle, so another publisher (dist, rotate,
        # circle, teleop) can move the car. A run and STOP are unaffected.
        self.declare_parameter('external_control', False)
        manual.declare_params(self)   # the tablet's movement buttons, see utils/manual.py
        # robot.wheelbase / steering_limit_* (URDF) + navigation.yaml, passed
        # by the launch; the same files are the defaults for a bare `ros2 run`.
        self.planner_settings = list(planner_params.to_flat(planner_params.load()).items())
        for name, value in self.planner_settings:
            self.declare_parameter(name, value)
        self.configure_planner()

        # (/cmd_vel, /run_status, the Foxglove drawings, /start_run ...: RunnerBase)
        self.create_subscription(String, '/obstacle_setup', self.setup_callback, 10)
        # Manual drive lives here, not in the bridge: this node already owns /cmd_vel.
        self.create_subscription(String, '/manual_drive', self.manual_drive_callback, 10)
        plan_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.plan_request_pub = self.create_publisher(String, '/plan_legs/request', plan_qos)
        self.create_subscription(String, '/plan_legs/result', self.plan_result_callback, plan_qos)
        self._session = uuid.uuid4().hex[:8]   # tells this runner's requests from an earlier one's
        self._remote = None                    # the request the laptop is planning, see plan()

        self.follower = PurePursuitController()
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
            self.get_logger().warn(f'Waiting for base_link -> camera_link (URDF): {exc}',
                                   throttle_duration_sec=2.0)
            return None
        return yaw_from_quaternion(tf.transform.rotation)

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
                    f"Reset not confirmed after {RESET_CONFIRM_TIMEOUT_S:.0f}s - pose {self.fmt(self.current_pose)} "
                    f"is not at the start {self.fmt(self.start_pose())}. Is the EKF running / did it take /set_pose?")

    def setup_callback(self, msg: String):
        """`id:x,y,facing|...` (metres). Replaces the previous set and plan, but
        never interrupts a run."""
        if self.in_run():
            self.get_logger().warn("Obstacle setup ignored: a run is in progress. Stop it first.")
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
                self.get_logger().warn(f"Skipping malformed obstacle entry {item!r}")
                continue
            if face not in markers.FACING:
                self.get_logger().warn(f"Skipping obstacle {obs_id!r}: facing {face!r} is not N/E/S/W")
                continue
            obstacles.append((x, y, face))
            ids.append(obs_id.strip())
        if not obstacles:
            self.get_logger().warn("Obstacle setup contained no valid obstacles - ignored.")
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
        """/start_run (tablet BEGIN, `pixi run go`): only from WAITING_FOR_GO with
        the plan done and the pose reset; otherwise says why not."""
        not_ready = {
            State.WAITING_FOR_SETUP: "Not ready: no obstacle setup received yet.",
            State.PLANNING_PATH: "Not ready: still planning the path.",
            State.STOPPED: "Ignored: stopped - reset the pose first (pixi run reset).",
            State.FINISHED: "Not ready: run finished - put the car back and reset (pixi run reset).",
        }
        if self.state == State.WAITING_FOR_GO and self.plan_state != 'DONE':
            reason = "Not ready: still planning the path."
        elif self.state == State.WAITING_FOR_GO and not self.reset_done:
            reason = "Not ready: reset the pose first (pixi run reset)."
        elif self.state != State.WAITING_FOR_GO:
            reason = not_ready.get(self.state, f"Ignored: run already in progress ({self.state.name}).")
        else:
            reason = None
        response.success = reason is None
        if reason:
            response.message = reason
            self.get_logger().warn(f"`go` rejected: {reason}")
        else:
            self.set_state(State.NAVIGATING_TO_TARGET)
            self.current_target_idx = 0
            self._manual_until = 0.0
            self.reset_done = False     # the car is leaving the start
            self.run_start, self.run_end = self.now(), None
            self._replanned = set()
            self._heading_retried = set()
            self._off_path = {}
            response.message = "Run started - navigating to targets."
            self.get_logger().info(f"GO        {' -> '.join(self.tablet_id(i) for i in self.visiting_order)}")
            self.publish_checkpoints()
        self.sync_indicators()
        return response

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
        self.follower.set_path([])
        if self.plan_state == 'PLANNING':
            self.plan_state = 'WAITING'   # a pose reset replans from the kept obstacles
        self.set_state(State.STOPPED)
        self.send_cmd(0.0, 0.0)
        self.end_run()
        self.get_logger().warn(f"STOP      at {self.fmt(self.current_pose)} after {self.run_time():.1f} s"
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
                self.set_state(State.WAITING_FOR_GO)
            elif self.obstacles:
                self._plan_gen += 1
                self.leg_paths = []
                self.plan_state = 'PLANNING'
                self.set_state(State.PLANNING_PATH)
            else:
                self.set_state(State.WAITING_FOR_SETUP)
        self.get_logger().info("RESET     ...")
        self.sync_indicators()

    def manual_drive_callback(self, msg: String):
        """A tablet movement button (utils/manual.py): a short burst on /cmd_vel,
        then zeros. Ignored during a run and while stopped."""
        key = msg.data.strip().lower()
        cmd = manual.burst(self, key, planner_params.ACTIVE.wheelbase)
        if cmd is None:
            self.get_logger().warn(f"Unknown manual drive command {msg.data!r}")
            return
        if self.stopped or self.in_run():
            self.get_logger().info(f"Manual drive {key!r} ignored (run in progress or stopped).")
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
        if self.state == State.WAITING_FOR_GO and self.plan_state == 'DONE' and self.reset_done:
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
            confirmed = (self.detected_target_id is not None
                         and Counter(self.scan_detections)[self.detected_target_id]
                         >= int(self.get_parameter('scan_confirm_count').value))
            if confirmed or now - self.state_start >= float(self.get_parameter('scan_pause_s').value):
                self.finish_scan()
        elif self.state == State.STOPPED:
            self.send_cmd(0.0, 0.0)
        elif self.state in (State.WAITING_FOR_GO, State.FINISHED):
            if not self.get_parameter('external_control').value:
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
        self.visiting_order, self.checkpoints, self.unreachable, self.costmap = plan_visiting_order(
            obstacles_cm, start, theta_offset=camera_yaw)
        self.publish_map()
        self.publish_checkpoints()
        if self.unreachable:
            self.get_logger().warn(
                f"Obstacles with no valid scan checkpoint, skipped: {[self.tablet_id(i) for i in self.unreachable]}")
        self.get_logger().info(f"PLAN      order {' -> '.join(self.tablet_id(i) for i in self.visiting_order)}")

        self.leg_paths = [None] * len(self.visiting_order)
        self.leg_starts = [None] * len(self.visiting_order)   # pose each leg was planned from
        self._replanned = set()
        self._heading_retried = set()
        self._off_path = {}
        self.current_target_idx = 0
        self.set_state(State.WAITING_FOR_GO)
        if self.get_parameter('remote_planner').value:
            self.request_remote_plan(obstacles_cm, start)
        else:
            self.plan_locally(start)

    def plan_locally(self, start):
        self._remote = None
        threading.Thread(target=self.plan_legs, daemon=True,
                         args=(start, self._plan_gen, self.leg_paths, self.checkpoints, self.costmap)).start()

    def request_remote_plan(self, obstacles_cm, start):
        """Ask task1_planner (laptop) for every leg; answers in plan_result_callback."""
        gen = f'{self._session}:{self._plan_gen}'
        self._remote = {'gen': gen, 'start': start, 'acked': False, 'last': self.now()}
        self.plan_request_pub.publish(String(data=json.dumps({
            'gen': gen, 'obstacles_cm': [list(o) for o in obstacles_cm], 'start': list(start),
            'checkpoints': [list(c) for c in self.checkpoints],
            'params': {name: self.get_parameter(name).value for name, _ in self.planner_settings}})))

    def plan_result_callback(self, msg: String):
        r = self._remote
        data = json.loads(msg.data)
        if r is None or data.get('gen') != r['gen']:
            return                       # an old request's answer
        r['last'] = self.now()
        if data.get('ack'):
            if not r['acked']:
                r['acked'] = True
                self.get_logger().info("PLAN      legs planned on the laptop")
            return
        idx = data['idx']
        self.leg_starts[idx] = tuple(data['start'])
        self.leg_paths[idx] = [tuple(p) for p in data['path']]
        if not data['path']:
            self.get_logger().warn(f"PLAN      leg {idx + 1}/{len(self.checkpoints)}: NO PATH")
        self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))
        if all(p is not None for p in self.leg_paths):
            self._remote = None
            self.plan_state = 'DONE'
            self.get_logger().info(f"PLAN      done, {len(self.checkpoints)} legs - waiting for GO")

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

    def plan_legs(self, start, gen, leg_paths, checkpoints, costmap):
        """Background thread: every leg back to back. Writes only single items of
        leg_paths (atomic under the GIL); control_loop polls them."""
        try:
            pose = start
            for idx, target in enumerate(checkpoints):
                if gen != self._plan_gen:
                    return    # abandoned (new setup / stop)
                path = plan_leg(costmap, pose, target, progress_callback=self.publish_search_progress)
                if gen != self._plan_gen:
                    return
                self.leg_starts[idx] = pose
                leg_paths[idx] = path
                if not path:
                    self.get_logger().warn(f"PLAN      leg {idx + 1}/{len(checkpoints)}: NO PATH")
                self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))
                pose = target
            if gen == self._plan_gen:
                self.plan_state = 'DONE'
                self.get_logger().info(f"PLAN      done, {len(checkpoints)} legs - waiting for GO")
        except Exception:
            self.get_logger().error(f"Background leg-planning thread crashed:\n{traceback.format_exc()}")

    def navigate(self):
        if self.current_target_idx >= len(self.leg_paths):
            self.send_cmd(0.0, 0.0)
            self.set_state(State.FINISHED)
            self.get_logger().info("FINISHED  nothing to visit")
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
        cmd = self.follower.compute_cmd()
        if cmd is None or self.follower.is_done():
            self.send_cmd(0.0, 0.0)
            idx = self.current_target_idx
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
            self.detected_target_id, self.scan_detections = None, []
            self.set_state(State.PAUSE_FOR_SCAN)
            self.get_logger().info(f"ARRIVED   #{self.current_label()} at {self.fmt(self.current_pose)} "
                                   f"(target {self.fmt((cx, cy, ct))}, heading {dyaw:+.0f}deg) - scanning")
        elif self.left_path():
            return
        else:
            self.send_cmd(*cmd)

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
            return True
        gears = [p[3] for p in path]
        cuts = [i for i in range(1, len(gears)) if gears[i] != gears[i - 1]]
        moves = ' '.join('fwd' if gears[a] >= 0 else 'REV' for a in [0] + cuts)
        self.get_logger().info(f"LEG {idx + 1}/{len(self.visiting_order)}   -> #{self.current_label()} "
                               f"{self.fmt(self.checkpoints[idx])}  {moves}")
        return True

    def off_leg_start(self, idx) -> bool:
        """Is the car further than REPLAN_POS_M / REPLAN_YAW_RAD from where leg idx
        was planned from? That pose, not the path's first point: the planner's
        path starts one step (5 cm) ahead of it, so every leg looked 5 cm off and
        was replanned for nothing (2026-10-01, up to 19 s per leg)."""
        x, y, yaw = self.current_pose
        px, py, ptheta = self.leg_starts[idx][:3]
        dyaw = abs(math.atan2(math.sin(yaw - ptheta), math.cos(yaw - ptheta)))
        return math.hypot(x - px, y - py) > REPLAN_POS_M or dyaw > REPLAN_YAW_RAD

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
            path = plan_leg(self.costmap, start, self.checkpoints[idx],
                            progress_callback=self.publish_search_progress)
            if gen != self._plan_gen:
                return
            self.leg_starts[idx] = start
            self.leg_paths[idx] = path
            self.path_pub.publish(markers.route_path(self.leg_paths, self.stamp()))
        except Exception:
            self.get_logger().error(f"Leg replanning thread crashed:\n{traceback.format_exc()}")
            self.leg_paths[idx] = []

    def finish_scan(self):
        """Report the most frequent detection, then go on to the next obstacle."""
        target_id = self.detected_target_id or "UNKNOWN"
        obs = self.tablet_id(self.visiting_order[self.current_target_idx])
        self.send_bt(f"TARGET,{obs},{target_id}")
        seen = ' '.join(f"{k}x{n}" for k, n in Counter(self.scan_detections).most_common()) or 'nothing'
        self.get_logger().info(f"TARGET    #{obs} = {target_id}   (YOLO saw {seen}, "
                               f"{self.now() - self.state_start:.1f} s)")

        self.current_target_idx += 1
        if self.current_target_idx >= len(self.visiting_order):
            self.set_state(State.FINISHED)
            self.end_run()
            self.get_logger().info(f"FINISHED  all obstacles visited in {self.run_time():.1f} s")
        else:
            self.set_state(State.NAVIGATING_TO_TARGET)   # the next tick loads the leg
        self.publish_checkpoints()

    # -------------------------------------------------------------- output ----

    def publish_map(self):
        stamp = self.stamp()
        self.grid_pub.publish(markers.costmap_grid(self.costmap, stamp))
        self.arena_pub.publish(markers.arena_markers(stamp))
        self.obstacle_pub.publish(markers.obstacle_markers(
            self.obstacles, [self.tablet_id(i) for i in range(len(self.obstacles))], stamp))

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
        self.run_status_pub.publish(msg)


def main(args=None):
    run(Task1Runner, args=args)


if __name__ == '__main__':
    main()
