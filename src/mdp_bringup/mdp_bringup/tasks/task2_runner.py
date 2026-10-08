"""
Task 2 runner: out of the carpark, past obstacle 1 on its arrow's side, round
obstacle 2 on its arrow's side, back into the carpark - as fast as it can.

The distances (obstacle 1 at 0.6-1.5 m, obstacle 2 another 0.6-1.5 m) are not
known in advance, so the car measures them with the front ultrasonic:

  Four checkpoints: 1 beside obstacle 1 (arrow 1's side), 2 beside obstacle 2's
  end on arrow 2's side, 3 beside its other end (round the back), 4 = home.

  WAITING_FOR_GO   standing in the carpark: the ultrasonic measures obstacle 1,
                   YOLO reads arrow 1 -> checkpoint 1 (heading as at home) and the
                   path to it are planned before GO
  TO_CHECKPOINT_1  GO: that path - turn out early, a straight diagonal, straighten
                   up at checkpoint 1. Arriving (pointing straight, level with
                   obstacle 1) the ultrasonic measures obstacle 2 (a wide bar on the
                   centre line) and YOLO reads arrow 2
  CHECKPOINT_1     both known -> checkpoints 2, 3, 4 and the smooth path through
                   them, planned right there (geometry, ~1 ms; stops only if a
                   reading is still missing or Hybrid A* is needed)
  FOLLOW           1 -> 2 diagonal, straight into 2; 2 -> 3 half a circle round the
                   back of the bar; 3 -> 4 diagonal onto the centre line, straight in
  (fallback, arrow 1 not read at home: straight on until YOLO reads it, then the
   same diagonal to checkpoint 1 from there - APPROACH_1; too close for that:
   SWERVE_OUT/BACK, full lock out and back)
  FINISHED         in the carpark (or the ultrasonic sees the back wall)

Frame (`map`): the task 2 course, x along it, centre line y = 1.2 m - see
utils/obstacle_layout.py. The course rules (carpark, obstacle sizes) come from
config/tasks.yaml task2; its `sim` layout is NOT read here. Settings:
config/navigation.yaml task2_runner. Drawn in Foxglove like task 1.
"""

import math
import statistics
import threading
import time
import traceback
from collections import Counter
from enum import Enum, auto

import numpy as np
import rclpy.time
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from mdp_interfaces.msg import RunStatus
from sensor_msgs.msg import Range
from std_msgs.msg import String

from mdp_algorithm.control.path_follower import PathFollower
from mdp_algorithm.planning.costmap import Costmap, Obstacle
from mdp_algorithm.planning.planner import plan_leg
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils import config, markers, obstacle_layout
from mdp_bringup.utils.obstacle_layout import TASK2_AREA_M, TASK2_BACK_WALL_X, TASK2_CENTRE_Y
from mdp_bringup.tasks.runner_base import RunnerBase
from mdp_bringup.utils.run import run

ARROWS = {'39': 'LEFT', 'LEFT': 'LEFT', '38': 'RIGHT', 'RIGHT': 'RIGHT'}   # YOLO id -> side
SIDE = {'LEFT': 1.0, 'RIGHT': -1.0}                                       # +y = left
HEADING_GAIN = 1.5            # rad of steering per rad of heading error, on straights
ARC_LEAD_RAD = math.radians(5.0)   # end an arc this early: the steering takes a moment to swing back
ARROW_WAIT_S = 3.0            # an arrow not read yet when needed: wait this long, then guess
US_VALID = (0.25, 2.2)        # m - lane readings outside this are not obstacle 2


class State(Enum):
    WAITING_FOR_GO = auto()
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
        layout = self.declare_parameter(
            'layout', f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml").value
        self.arena = obstacle_layout.load_task2(layout)
        self.car = planner_params.ACTIVE          # URDF + navigation.yaml (robot.*, follower.*)
        car_centre_ahead = (self.car.footprint_front - self.car.footprint_rear) / 2.0
        self.home = self.arena.car_pose_centred(car_centre_ahead, math.pi)

        # (/cmd_vel, /run_status, the Foxglove drawings, /start_run ...: RunnerBase)
        self.create_subscription(Range, '/ultrasonic', self.ultrasonic_callback, 10)

        self.follower = PathFollower()
        # Task 2 sets the speed and lookahead itself (speed_ahead(), follower_step()).
        self.follower.regulate = self.follower.velocity_scaled_lookahead = False
        self.us_offset = None            # base_link -> ultrasonic_link, m ahead (TF, from the URDF)
        self.reset_run()
        self.publish_map()
        self.get_logger().info("READY     waiting for GO")

    def reset_run(self):
        self.set_state(State.WAITING_FOR_GO)
        self.go_received = False
        self.run_start = self.run_end = None
        self.current_pose = (0.0, 0.0, 0.0)
        self.have_pose = False           # GO waits for the first EKF pose
        self.us_range = None             # (range m, time s) - latest valid ultrasonic reading
        self.us_trigger_count = 0
        self.arrow_counts = [Counter(), Counter()]   # per obstacle
        self.arrows = [None, None]
        self._arrow_wait_since = None
        self.side = 0.0                  # +1 left / -1 right, the lane beside obstacle 1
        self.x1 = self.x2 = None         # obstacle centres (m, along the course)
        self.lane_readings = []
        self.waypoints = []              # (x, y, theta): A, B, HOME
        self.names = ['checkpoint 2', 'checkpoint 3', 'checkpoint 4 (home)']
        self.leg_paths = []
        self.leg = 0
        self.costmap = None
        self.pre = None                  # the path home -> beside obstacle 1, planned before GO
        self.lane_waypoint = None        # checkpoint 1, beside obstacle 1
        self.x1_seen = None              # obstacle 1 as the ultrasonic sees it from home (drawn)
        self._plan_gen = getattr(self, '_plan_gen', 0) + 1
        self.follower.set_path([])

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

    def ultrasonic_offset(self) -> float:
        """How far ahead of base_link the ultrasonic sits (URDF, via TF)."""
        if self.us_offset is None:
            try:
                tf = self.tf_buffer.lookup_transform('base_link', 'ultrasonic_link', rclpy.time.Time())
                self.us_offset = tf.transform.translation.x
            except tf2_ros.TransformException:
                self.get_logger().warn("PLAN      no base_link -> ultrasonic_link transform - assuming 0.19 m",
                                       throttle_duration_sec=5.0)
                return 0.19
        return self.us_offset

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

    def arrow(self, i: int):
        counts = self.arrow_counts[i]
        return counts.most_common(1)[0][0] if counts else None

    def need_arrow(self, i: int) -> bool:
        """Arrow i known -> True. Not yet: stop, wait up to ARROW_WAIT_S, then guess."""
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
        self.arrows[i] = 'LEFT' if i == 0 else 'RIGHT'
        self._arrow_wait_since = None
        self.get_logger().warn(f"ARROW {i + 1}   never read - guessing {self.arrows[i]}")
        return True

    # ---------------------------------------------------------- callbacks ----

    def ultrasonic_callback(self, msg: Range):
        if math.isfinite(msg.range) and msg.min_range <= msg.range <= msg.max_range:
            self.us_range = (float(msg.range), self.now())

    def yolo_callback(self, msg: String):
        side = ARROWS.get(msg.data.strip().upper())
        if side is None:
            return
        # Arrow 1 until the car turns out for checkpoint 1; arrow 2 once it is level
        # with obstacle 1 (its face out of view) and at checkpoint 1.
        if self.state in (State.WAITING_FOR_GO, State.APPROACH_1):
            i = 0
        elif self.state == State.CHECKPOINT_1 or (self.state == State.TO_CHECKPOINT_1 and self.level_with_1()):
            i = 1
        else:
            return
        if self.arrows[i] is None:
            if not self.arrow_counts[i]:
                self.get_logger().info(f"ARROW {i + 1}   sees {side}")
            self.arrow_counts[i][side] += 1

    def start_run_callback(self, request, response):
        response.success = self.state == State.WAITING_FOR_GO and not self.go_received and self.have_pose
        if not self.have_pose:
            response.message = "Not ready: no pose from the EKF yet."
            self.get_logger().warn(f"`go` rejected: {response.message}")
            return response
        if response.success:
            self.go_received = True
            response.message = "Run started - task 2."
        else:
            response.message = f"Ignored: {self.state.name} - reset (pixi run reset) to run again."
            self.get_logger().warn(f"`go` rejected: {response.message}")
        return response

    def stop_run_callback(self, request, response):
        self._plan_gen += 1
        self.follower.set_path([])
        self.set_state(State.STOPPED)
        self.send_cmd(0.0, 0.0)
        self.end_run()
        self.get_logger().info(f"STOP      at {self.fmt(self.current_pose)} after {self.run_time():.1f} s"
                               f" - reset before the next run")
        response.success, response.message = True, "Stopped."
        return response

    def set_pose_callback(self, msg):
        """A pose reset (car back in the carpark) readies the next run; during a
        run it stops it first."""
        if self.state not in (State.FINISHED, State.STOPPED, State.WAITING_FOR_GO):
            self.send_cmd(0.0, 0.0)
            self.get_logger().warn("RESET     during a run - stopping it")
        self.reset_run()
        self.publish_map()
        self.get_logger().info("RESET     ready - waiting for GO")

    # -------------------------------------------------------------- control ----

    def control_loop(self):
        handler = {
            State.WAITING_FOR_GO: self.waiting,
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
        if not self.go_received:
            self.show_obstacle_1()
            self.preplan()
            return
        self.run_start = self.now()
        pre = self.pre
        if pre and pre.get('path'):
            # Planned at home: straight to the waypoint beside obstacle 1.
            self.arrows[0], self.x1, self.side = pre['arrow'], pre['x1'], SIDE[pre['arrow']]
            self.lane_waypoint = pre['goal']
            self.follower.set_path(pre['path'])
            self.get_logger().info(f"GO        arrow 1 {pre['arrow']}, obstacle 1 at "
                                   f"({markers.cell(self.x1)},{markers.cell(TASK2_CENTRE_Y)}) -> beside it "
                                   f"{self.fmt(pre['goal'])}")
            self.publish_map()
            self.set_state(State.TO_CHECKPOINT_1)
            self.publish_waypoints()
            return
        self.get_logger().info(f"GO        from {self.fmt(self.current_pose)}, arrow 1 not read at home - "
                               f"driving on until it is")
        self.set_state(State.APPROACH_1)

    def show_obstacle_1(self):
        """At home: draw obstacle 1 and its costmap as soon as the ultrasonic reads it."""
        r = self.fresh_range()
        if r is None or not (US_VALID[0] <= r <= US_VALID[1]):
            return
        x1 = self.ahead_x(r, self.arena.obstacle_1_size[0])
        if self.x1_seen is None or abs(self.x1_seen - x1) > 0.03:
            self.x1_seen = x1
            self.costmap = self.make_costmap(x1, None)
            self.publish_map()

    def preplan(self):
        """Before GO: once the ultrasonic sees obstacle 1 and YOLO has read arrow 1,
        plan home -> beside obstacle 1 (again if either changes)."""
        arrow, r = self.arrow(0), self.fresh_range()
        if arrow is None or r is None or not (US_VALID[0] <= r <= US_VALID[1]):
            return
        x1 = self.ahead_x(r, self.arena.obstacle_1_size[0])
        pre = self.pre
        if pre and (pre.get('planning') or (pre['arrow'] == arrow and abs(pre['x1'] - x1) < 0.03)):
            return
        goal = (x1, TASK2_CENTRE_Y + SIDE[arrow] * float(self.p('lane_offset')), 0.0)
        self.pre = pre = {'arrow': arrow, 'x1': x1, 'goal': goal, 'path': None, 'planning': True}
        costmap = self.make_costmap(x1, None)

        def plan(start=tuple(self.current_pose)):
            pre['path'] = (self.lined_up_path(start, goal, costmap, float(self.p('line_up_dist')))
                           or plan_leg(costmap, start, goal))
            pre['planning'] = False
            if pre['path']:
                self.lane_waypoint = goal
                self.path_pub.publish(markers.route_path([pre['path']], self.stamp()))
                self.publish_waypoints()
                self.get_logger().info(f"PLAN      home -> beside obstacle 1 {self.fmt(goal)} (arrow 1 {arrow})"
                                       f" - ready for GO")
        threading.Thread(target=plan, daemon=True).start()

    def geometric_leg(self, idx, start, target, costmap):
        """Leg idx as clean geometry, or None: 1 -> 2 a diagonal, straight into
        checkpoint 2; 2 -> 3 half a circle round the bar; 3 -> 4 a diagonal onto the
        centre line, straight into the carpark."""
        if idx == 1:
            return self.round_path(start, target, costmap)
        # Tight layouts: less straight-in and less spare room before giving up on
        # clean geometry (Hybrid A* makes the car wait while it plans).
        if idx == 0:
            straight, margin = float(self.p('line_up_dist')), 0.04
        else:
            straight, margin = float(self.p('home_straight_dist')), float(self.p('home_margin'))
        for f_straight, f_margin in ((1.0, 1.0), (0.5, 1.0), (0.5, 0.5), (0.0, 0.5)):
            path = self.lined_up_path(start, target, costmap, straight * f_straight, margin * f_margin)
            if path:
                return path
        return None

    def geometric_legs(self, start, waypoints, costmap):
        paths, pose = [], start
        for idx, target in enumerate(waypoints):
            paths.append(self.geometric_leg(idx, pose, target, costmap))
            pose = target
        return paths

    def round_path(self, a, b, costmap, step=0.02):
        """A -> B round the back of obstacle 2: half a circle through both (they
        face opposite ways, level with the bar's centre). None if it doesn't fit."""
        r = abs(b[1] - a[1]) / 2.0
        if r < max(self.car.minimum_turning_radius_left, self.car.minimum_turning_radius_right) * \
                self.car.turning_radius_margin:
            return None
        cx, cy = a[0], (a[1] + b[1]) / 2.0
        turn = 1.0 if b[1] > a[1] else -1.0          # left (counter-clockwise) or right round the back
        path = [(cx + r * math.sin(t), cy - turn * r * math.cos(t), a[2] + turn * t, 1)
                for t in np.arange(0.0, math.pi, step / r)] + [(b[0], b[1], b[2], 1)]
        return path if self.clear(path, costmap) else None

    def lined_up_path(self, start, goal, costmap, straight, margin=0.04):
        """Start -> goal (same heading) as one diagonal that is straight again
        `straight` before the goal, then straight in: the car is parallel to
        whatever is beside the goal when it gets there (B -> HOME: into the
        carpark; L -> A: along the side wall). None if it doesn't fit."""
        entry = (goal[0] - straight * math.cos(goal[2]), goal[1] - straight * math.sin(goal[2]), goal[2])
        path = self.diagonal_path(start, entry, costmap, margin)
        if not path:
            return None
        n = max(1, int(straight / 0.02))
        tail = [(entry[0] + (goal[0] - entry[0]) * k / n, entry[1] + (goal[1] - entry[1]) * k / n, goal[2], 1)
                for k in range(1, n + 1)]
        if not self.clear(tail, costmap, margin):
            return None
        return path + tail

    @staticmethod
    def clear(path, costmap, margin=0.04) -> bool:
        """Every (3rd) pose collision-free, and still so shifted `margin` to either
        side: room for the car running a few cm off the line."""
        for x, y, th, _ in path[::3]:
            nx, ny = -math.sin(th) * margin, math.cos(th) * margin
            for ox, oy in ((0.0, 0.0), (nx, ny), (-nx, -ny)):
                if costmap.in_collision((x + ox) * 100.0, (y + oy) * 100.0, th):
                    return False
        return True

    CURVE_SCALES = (6.0, 4.0, 3.0, 2.0, 1.5, 1.0)   # x the smallest planned turning circle

    def diagonal_path(self, start, goal, costmap, margin=0.04, step=0.02):
        """Start -> goal (same heading) as the FASTEST of these shapes that fits:
        straight, a curve out, a straight diagonal, a curve back - tried with
        curves from wide (fast: curve speed = sqrt(max_lateral_accel x radius))
        down to the smallest planned turning circle, each turning out as early as
        it fits. Poses (x, y, theta, gear) every `step` m, or None if nothing fits
        with `margin` of room either side - then Hybrid A* plans instead."""
        # The start pose's frame: start at the origin facing +x (goal has the same heading).
        c, s_ = math.cos(start[2]), math.sin(start[2])
        gx, gy = goal[0] - start[0], goal[1] - start[1]
        gx, gy = c * gx + s_ * gy, -s_ * gx + c * gy
        side = 1.0 if gy > 0 else -1.0
        k = self.car.turning_radius_margin
        r_left = self.car.minimum_turning_radius_left * k
        r_right = self.car.minimum_turning_radius_right * k
        best, best_t = None, float('inf')
        for scale in self.CURVE_SCALES:
            r1, r2 = (r_left * scale, r_right * scale) if side > 0 else (r_right * scale, r_left * scale)
            xs = 0.0
            while xs < gx:
                local = self._diagonal(0.0, 0.0, xs, gx - xs, abs(gy), side, r1, r2, step)
                if local:
                    path = [(start[0] + c * x - s_ * y, start[1] + s_ * x + c * y, start[2] + th, g)
                            for x, y, th, g in local]
                    if self.clear(path, costmap, margin):
                        t = self.drive_time(path)
                        if t < best_t:
                            best, best_t = path, t
                        break           # the earliest turn that fits, for this curve size
                xs += 0.05
        return best

    def drive_time(self, path) -> float:
        """Seconds to drive `path` at the speed profile (straight_speed, curves at
        sqrt(max_lateral_accel x radius))."""
        v_max, a = float(self.p('straight_speed')), float(self.p('max_lateral_accel'))
        t = 0.0
        for p0, p1 in zip(path, path[1:]):
            ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if ds < 1e-6:
                continue
            kappa = abs(math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))) / ds
            t += ds / (min(v_max, math.sqrt(a / kappa)) if kappa > 1e-6 else v_max)
        return t

    @staticmethod
    def _diagonal(x0, y0, xs, d, h, side, r1, r2, step):
        """Straight x0 -> xs, arc r1 up to angle a, straight L, arc r2 back to 0,
        ending d ahead of xs and h to the side. None if it can't be done."""
        R = r1 + r2
        amax = math.asin(min(1.0, d / R)) if d < R else math.pi / 2.0 - 1e-6

        def lateral(a):     # lateral offset reached with turn angle a (L from the forward distance)
            L = (d - R * math.sin(a)) / math.cos(a)
            return R * (1 - math.cos(a)) + L * math.sin(a), L
        if lateral(amax)[0] < h:
            return None
        lo, hi = 1e-6, amax
        for _ in range(60):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if lateral(mid)[0] < h else (lo, mid)
        a = (lo + hi) / 2.0
        L = lateral(a)[1]
        pts = [(x0 + t, y0, 0.0, 1) for t in np.arange(0.0, xs - x0, step)]
        for t in np.arange(0.0, a, step / r1):                                   # turn out
            pts.append((xs + r1 * math.sin(t), y0 + side * r1 * (1 - math.cos(t)), side * t, 1))
        px, py = xs + r1 * math.sin(a), y0 + side * r1 * (1 - math.cos(a))
        for t in np.arange(0.0, L, step):                                        # the diagonal
            pts.append((px + t * math.cos(a), py + side * t * math.sin(a), side * a, 1))
        px, py = px + L * math.cos(a), py + side * L * math.sin(a)
        cx, cy = px + r2 * math.sin(a), py - side * r2 * math.cos(a)
        for u in np.arange(0.0, a, step / r2):                                   # turn back
            pts.append((cx - r2 * math.sin(a - u), cy + side * r2 * math.cos(a - u), side * (a - u), 1))
        pts.append((cx, cy + side * r2, 0.0, 1))                                 # exactly the goal
        return [(float(x), float(y), float(th), g) for x, y, th, g in pts]

    def level_with_1(self) -> bool:
        """Beside obstacle 1 and pointing (nearly) straight: the ultrasonic and the
        camera now look past it at obstacle 2."""
        return (self.x1 is not None and self.current_pose[0] >= self.x1 - 0.10
                and abs(self.current_pose[2]) < math.radians(8.0))

    def read_obstacle_2(self):
        """One ultrasonic reading of obstacle 2 (its centre along the course), kept
        if it is beyond obstacle 1."""
        r = self.fresh_range(0.1)
        if r is not None and US_VALID[0] <= r <= US_VALID[1] and self.level_with_1():
            x2 = self.ahead_x(r, self.arena.obstacle_2_size[0])
            if x2 > self.x1 + 0.3:
                self.lane_readings.append(x2)

    def to_checkpoint_1(self):
        """The path from the carpark to checkpoint 1; obstacle 2 is measured in its
        last moments (pointing straight, level with obstacle 1)."""
        self.leg_pub.publish(markers.leg_markers(self.follower.path, self.follower.last_target,
                                                 self.current_pose, self.stamp()))
        self.read_obstacle_2()
        if not self.follower_step(stop_at_end=False):
            return
        # Keep rolling straight on while checkpoints 2-4 are planned (no stop).
        self.hold_heading(max(abs(self._last_cmd[0]), float(self.p('path_speed'))), 0.0)
        self.arrive_checkpoint_1()

    def arrive_checkpoint_1(self):
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
                self.follower.set_path(path)
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
        """At checkpoint 1: obstacle 2's distance (ultrasonic) and arrow 2 (YOLO)
        -> checkpoints 2, 3, 4 and the path through them, from right here. Waits,
        stopped, only for a reading still missing."""
        self.read_obstacle_2()
        if not self.lane_readings:
            self.send_cmd(0.0, 0.0)
            if self.now() - self.state_start > 1.5:
                self.get_logger().error("CHECKPOINT 1  obstacle 2 not seen by the ultrasonic - stopping")
                self.set_state(State.STOPPED)
            return
        if not self.need_arrow(1):
            return
        self.x2 = statistics.median(self.lane_readings)
        s2 = SIDE[self.arrows[1]]
        half2 = self.arena.obstacle_2_size[1] / 2.0 + float(self.p('side_clearance'))
        cy = TASK2_CENTRE_Y
        self.waypoints = [(self.x2, cy + s2 * half2, 0.0), (self.x2, cy - s2 * half2, math.pi), self.home]
        # Plan from where the car will be once the plan is ready (it rolls on meanwhile).
        x, y, _ = self.current_pose
        start = (x + abs(self._last_cmd[0]) * 0.15, y, 0.0)
        self.get_logger().info(f"OBSTACLE 2 at ({markers.cell(self.x2)},{markers.cell(cy)}), arrow 2 {self.arrows[1]}"
                               f" -> checkpoint 2 {self.fmt(self.waypoints[0])}  3 {self.fmt(self.waypoints[1])}"
                               f"  4 {self.fmt(self.home)}")
        self.build_costmap()
        self.publish_map()
        self.leg, self.plan_start = 0, start
        self.set_state(State.FOLLOW)
        self.publish_waypoints()
        # The clean geometric legs take ~1 ms: plan them here and keep going.
        paths = self.geometric_legs(start, self.waypoints, self.costmap)
        if all(paths):
            self.leg_paths = paths
            self.path_pub.publish(markers.route_path(paths, self.stamp()))
            self.get_logger().info("PLAN      checkpoints 2, 3, 4 - smooth path")
            return
        # Some leg doesn't fit: Hybrid A* for it, in the background (the car waits).
        self.leg_paths = [None] * len(self.waypoints)
        threading.Thread(target=self.plan_legs, daemon=True,
                         args=(start, self._plan_gen, self.leg_paths, list(self.waypoints), self.costmap)).start()

    def follow(self):
        """Drive the planned legs; until leg 1 is ready, keep straight to its start."""
        if not self.follower.active:
            path = self.leg_paths[self.leg] if self.leg < len(self.leg_paths) else []
            if path is None:                   # still planning (Hybrid A*): wait
                self.send_cmd(0.0, 0.0)
                return
            if not path:
                self.send_cmd(0.0, 0.0)
                self.get_logger().error(f"LEG {self.leg + 1}/3   NO PATH to {self.names[self.leg]} - stopping")
                self.set_state(State.STOPPED)
                return
            self.follower.set_path(path)
            gears = [pt[3] for pt in path]
            moves = ' '.join('fwd' if gears[i] >= 0 else 'REV'
                             for i in range(len(gears)) if i == 0 or gears[i] != gears[i - 1])
            self.get_logger().info(f"LEG {self.leg + 1}/3   -> {self.names[self.leg]} "
                                   f"{self.fmt(self.waypoints[self.leg])}  {moves}")
            self.publish_waypoints()
        # Into the carpark: the ultrasonic sees the back wall -> done.
        r = self.fresh_range()
        if (self.leg == len(self.waypoints) - 1 and r is not None and r <= float(self.p('home_stop_dist'))
                and self.current_pose[0] < self.arena.opening_x):
            return self.finish("the back wall")
        self.publish_leg()
        if self.follower_step():
            self.get_logger().info(f"ARRIVED   {self.names[self.leg]} at {self.fmt(self.current_pose)}")
            self.leg += 1
            self.follower.set_path([])
            if self.leg >= len(self.waypoints):
                self.finish("HOME")

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
        """Speed for the path ahead (from the car, slow_down_dist plus the lookahead):
        as fast as its tightest curve allows at max_lateral_accel (v = sqrt(a * R)),
        between path_speed and straight_speed. A gear change or the end of the
        path in that stretch: path_speed."""
        path = self.follower.path
        v_min, v_max = float(self.p('path_speed')), float(self.p('straight_speed'))
        if not path:
            return v_min
        x, y, _ = self.current_pose
        i0 = max(0, self.follower._search_idx - 40)
        k = min(range(i0, min(len(path), self.follower._search_idx + 1)),
                key=lambda j: (path[j][0] - x) ** 2 + (path[j][1] - y) ** 2)
        window = float(self.p('slow_down_dist')) + self.follower.lookahead_dist
        kappa, dist = 0.0, 0.0
        for j in range(k + 1, len(path)):
            ds = math.hypot(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1])
            dist += ds
            if path[j][3] != path[k][3]:
                return v_min
            if ds > 1e-4:
                dth = math.atan2(math.sin(path[j][2] - path[j - 1][2]), math.cos(path[j][2] - path[j - 1][2]))
                kappa = max(kappa, abs(dth) / ds)
            if dist >= window:
                break
        else:
            return v_min            # the end of the path is within the window
        v = math.sqrt(float(self.p('max_lateral_accel')) / kappa) if kappa > 1e-6 else v_max
        return max(v_min, min(v_max, v))

    def finish(self, why: str):
        self.send_cmd(0.0, 0.0)
        self.follower.set_path([])
        self.set_state(State.FINISHED)
        self.end_run()
        self.get_logger().info(f"FINISHED  in the carpark at {self.fmt(self.current_pose)} ({why}) "
                               f"in {self.run_time():.1f} s")
        self.publish_waypoints()

    # -------------------------------------------------------------- planning ----

    def build_costmap(self):
        self.costmap = self.make_costmap(self.x1, self.x2)

    def make_costmap(self, x1, x2):
        """The course as far as it is known: carpark walls, the obstacles measured
        so far, and the side walls that MAY stand beside obstacle 2 (assumed there)."""
        cm = 100.0
        s1, s2 = self.arena.obstacle_1_size, self.arena.obstacle_2_size
        obstacles = [Obstacle(x1 * cm, TASK2_CENTRE_Y * cm, 'W', 0, s1[0] * cm, s1[1] * cm)]
        if x2 is not None:
            obstacles.append(Obstacle(x2 * cm, TASK2_CENTRE_Y * cm, 'W', 1, s2[0] * cm, s2[1] * cm))
        walls = self.arena.carpark_walls() + (self.arena.side_walls(x2) if x2 is not None else [])
        return Costmap(obstacles, arena_cm=(TASK2_AREA_M[0] * cm, TASK2_AREA_M[1] * cm),
                       walls=[tuple(v * cm for v in w) for w in walls],
                       edge_padding_cm=planner_params.ACTIVE.footprint_padding * cm)   # real walls

    def plan_legs(self, start, gen, leg_paths, waypoints, costmap):
        """Background thread: every leg back to back (like task 1)."""
        try:
            t0 = time.monotonic()
            pose = start
            for idx, target in enumerate(waypoints):
                # Clean geometry first (fast to drive); the planner only if it doesn't fit.
                path = self.geometric_leg(idx, pose, target, costmap)
                if not path:
                    path = plan_leg(costmap, pose, target, progress_callback=self.publish_search_progress)
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
        for i, (x, size) in enumerate(((x1, self.arena.obstacle_1_size), (self.x2, self.arena.obstacle_2_size))):
            if x is not None:
                blocks.append((x, cy, 'W'))
                sizes.append(size[:2])
                labels.append(str(i + 1))
        walls = self.arena.carpark_walls() + (self.arena.side_walls(self.x2) if self.x2 is not None else [])
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
        planning = self.leg_paths and any(p is None for p in self.leg_paths)
        msg.plan_state = 'PLANNING' if planning else ('DONE' if self.leg_paths else 'WAITING')
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
    run(Task2Runner, args=args)


if __name__ == '__main__':
    main()
