#!/usr/bin/env python3
"""Approach an obstacle on the front ultrasonic, then drive one full square
loop around it so the left-facing camera sees all 4 faces for YOLO.

WHY THIS EXISTS: neither drive_distance.py nor rotate_angle.py know about the
obstacle - they are the building blocks (closed-loop straight leg, closed-loop
turn), this script is the behaviour on top. It does two things:

1. APPROACH - drive forward watching /ultrasonic (front-facing HC-SR04) and
   stop once the obstacle is within --stop-cm. This phase is its own small
   node because it is the only part that needs the ultrasonic reading;
   nothing else here does.

2. LOOP - the camera faces LEFT (90 deg off chassis-forward), not forward, so
   stopping square-on to the obstacle shows the camera nothing useful - it is
   looking out to the side, past the obstacle. To get all 4 faces in view,
   drive a full square around it: turn --turn-deg (default 90) the SAME
   direction 4 times, with a straight --leg-m leg driven between each turn.
   Shelling out to rotate_angle.py and drive_distance.py for this reuses
   their closed-loop stopping (EKF yaw / wheel odometry) instead of
   duplicating it here.

   Direction defaults to LEFT. Picture driving laps on an oval track: turning
   left the whole way round keeps the infield on the driver's left the whole
   time. Same here - the obstacle is the "infield", the camera looks left, so
   a left-turning (CCW, viewed from above) loop keeps the obstacle on the
   camera's side for the entire lap. --direction right sweeps the other way,
   for a chassis where that mapping is flipped.

Usage:
    ros2 run mdp_bringup circle_obstacle.py
    ros2 run mdp_bringup circle_obstacle.py --stop-cm 30 --leg-m 0.6
    ros2 run mdp_bringup circle_obstacle.py --direction right
    pixi run circle
"""

import argparse
import math
import subprocess
import sys

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Range

CMD_TOPIC = '/cmd_vel'
ULTRASONIC_TOPIC = '/ultrasonic'
PUBLISH_HZ = 20.0


class ApproachObstacle(Node):
    """Drive straight forward until the front ultrasonic reports the
    obstacle within stop_m, or give up after timeout_s with nothing seen."""

    def __init__(self, stop_m: float, speed_mps: float, timeout_s: float):
        super().__init__('approach_obstacle')

        self.stop_m = stop_m
        self.speed_mps = speed_mps
        self.timeout_s = timeout_s

        self.last_range_m = math.inf
        self.have_reading = False
        self.finished = False
        self.reached = False
        self.start_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, 10)
        self.create_subscription(
            Range, ULTRASONIC_TOPIC, self._on_range,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        self.get_logger().info(
            f'approaching obstacle, stopping at {self.stop_m * 100:.0f} cm, '
            f'watching {ULTRASONIC_TOPIC}')

    def _on_range(self, msg: Range) -> None:
        self.last_range_m = msg.range
        self.have_reading = True

    def _publish(self, vx: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx
        self.cmd_pub.publish(msg)

    def _tick(self) -> None:
        if self.finished:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if not self.have_reading:
            self._publish(0.0)
            if elapsed > 5.0:
                self._stop(False, f'no reading on {ULTRASONIC_TOPIC} after 5s - '
                           f'is bringup running?')
            return

        if math.isfinite(self.last_range_m) and self.last_range_m <= self.stop_m:
            self._stop(True, f'obstacle at {self.last_range_m * 100:.1f} cm '
                       f'(threshold {self.stop_m * 100:.0f} cm)')
            return

        if elapsed > self.timeout_s:
            self._stop(False, f'TIMEOUT after {elapsed:.1f}s - nothing within '
                       f'{self.stop_m * 100:.0f} cm (last reading '
                       f'{self.last_range_m * 100:.1f} cm)')
            return

        self._publish(self.speed_mps)

    def _stop(self, reached: bool, reason: str) -> None:
        self.finished = True
        self.reached = reached
        for _ in range(5):
            self._publish(0.0)
        self.get_logger().info(reason)


def approach(stop_cm: float, speed_mps: float, timeout_s: float):
    """Returns (reached, final_range_m). The range is re-read after the car has
    actually stopped, since it coasts a little past the trigger distance."""
    node = ApproachObstacle(stop_cm / 100.0, speed_mps, timeout_s)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(15):
            rclpy.spin_once(node, timeout_sec=0.05)
        return node.reached, node.last_range_m
    finally:
        node.destroy_node()


# ----------------------------------------------------------------------------
# PATH PLANNER (added 2026-09-21). The original script did 4 x (turn, leg)
# from the stop point. Simulated with the bicycle model that loop lies BESIDE
# the obstacle, not around it, and passes ~13 cm from a 10 cm cube's centre.
# This plans instead: turn away (opposite --direction) first so the obstacle
# ends up on the camera's side, then loop around it, and CHECKS the result
# (car footprint clearance, and that the left-facing camera axis crosses all 4
# faces) before driving anything.
# Geometry frame: x forward-right, y forward, origin = rear axle at the stop
# point, heading +y. Camera faces LEFT of the heading.
# ----------------------------------------------------------------------------
WHEELBASE_M = 0.1433
STEER_DEG = 28.0       # rotate_angle.py's default lock
ARC_R = WHEELBASE_M / math.tan(math.radians(STEER_DEG))


def _arc(p, turn_deg):
    x, y, h = p
    sgn = 1.0 if turn_deg > 0 else -1.0
    a = math.radians(abs(turn_deg))
    cx = x - sgn * ARC_R * math.sin(h)
    cy = y + sgn * ARC_R * math.cos(h)
    h2 = h + sgn * a
    return (cx + sgn * ARC_R * math.sin(h2), cy - sgn * ARC_R * math.cos(h2), h2)


def _fwd(p, d):
    x, y, h = p
    return (x + d * math.cos(h), y + d * math.sin(h), h)


def sample_path(steps, start=(0.0, 0.0, math.pi / 2)):
    """steps: [('turn', deg_left_positive) | ('leg', metres)] -> list of poses."""
    pts = [start]
    p = start
    for kind, v in steps:
        n = 30 if kind == 'turn' else 20
        for i in range(1, n + 1):
            pts.append(_arc(p, v * i / n) if kind == 'turn' else _fwd(p, v * i / n))
        p = _arc(p, v) if kind == 'turn' else _fwd(p, v)
    return pts


def _rect_dist(px, py, half, cx, cy):
    dx = max(abs(px - cx) - half, 0.0)
    dy = max(abs(py - cy) - half, 0.0)
    return math.hypot(dx, dy)


def footprint_clearance(pose, obs, geom):
    """Min distance (m) from the cube to the car body at this pose (sampled)."""
    x, y, h = pose
    ox, oy, half = obs
    best = 9.0
    fx, fy = math.cos(h), math.sin(h)          # forward
    lx, ly = -math.sin(h), math.cos(h)         # left
    for u in (-geom['rear'], 0.0, geom['front']):
        for v in (-geom['half_w'], 0.0, geom['half_w']):
            px = x + u * fx + v * lx
            py = y + u * fy + v * ly
            best = min(best, _rect_dist(px, py, half, ox, oy))
    return best


def face_views(pts, obs, geom, min_view_m, max_view_m):
    """For each of the 4 faces, the smallest camera-to-face distance (within
    [min_view_m, max_view_m]) along the camera's left-facing axis over the whole
    path (None if the face is never seen from a usable distance)."""
    ox, oy, half = obs
    faces = {  # name: (segment a, segment b, outward normal)
        'near': ((ox - half, oy - half), (ox + half, oy - half), (0.0, -1.0)),
        'right': ((ox + half, oy - half), (ox + half, oy + half), (1.0, 0.0)),
        'far': ((ox + half, oy + half), (ox - half, oy + half), (0.0, 1.0)),
        'left': ((ox - half, oy + half), (ox - half, oy - half), (-1.0, 0.0)),
    }
    seen = {k: None for k in faces}
    for x, y, h in pts:
        fx, fy = math.cos(h), math.sin(h)
        lx, ly = -math.sin(h), math.cos(h)
        cx = x + geom['cam_fwd'] * fx + geom['half_w'] * lx
        cy = y + geom['cam_fwd'] * fy + geom['half_w'] * ly
        for name, ((ax, ay), (bx, by), (nx, ny)) in faces.items():
            if nx * lx + ny * ly >= 0.0:      # face turned away from the camera
                continue
            ex, ey = bx - ax, by - ay
            den = lx * ey - ly * ex
            if abs(den) < 1e-9:
                continue
            t = ((ax - cx) * ey - (ay - cy) * ex) / den      # along camera axis
            u = ((ax - cx) * ly - (ay - cy) * lx) / den      # along the face
            if 0.0 <= u <= 1.0 and min_view_m <= t <= max_view_m:
                if seen[name] is None or t < seen[name]:
                    seen[name] = t
    return seen


def plan_loop(direction, stop_range_m, obstacle_m, geom, min_view_m, max_view_m,
              margin_m):
    """Search legs (a, b, c) for: turn away, leg a, then 4 turns toward the
    obstacle with legs b, c, b, c. Returns the best (steps, report) or None."""
    sgn = 1.0 if direction == 'left' else -1.0
    ox = 0.0
    oy = stop_range_m + geom['front'] + obstacle_m / 2.0
    obs = (ox, oy, obstacle_m / 2.0)
    best = None
    for a in (0.0, 0.1, 0.2, 0.3):
        for b in (0.0, 0.1, 0.2, 0.3, 0.4):
            for c in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
                # obstacle must be on the CAMERA (left) side: turn away first
                # (right), then loop turning left. For --direction right the
                # frame is mirrored.
                steps = [('turn', -90.0 * sgn), ('leg', a),
                         ('turn', 90.0 * sgn), ('leg', b),
                         ('turn', 90.0 * sgn), ('leg', c),
                         ('turn', 90.0 * sgn), ('leg', b),
                         ('turn', 90.0 * sgn), ('leg', c)]
                pts = sample_path(steps)
                if sgn < 0:   # mirror x so the obstacle test stays in one frame
                    pts_eval = [(-x, y, math.pi - h) for x, y, h in pts]
                else:
                    pts_eval = pts
                clear = min(footprint_clearance(q, obs, geom) for q in pts_eval)
                if clear < margin_m:
                    continue
                views = face_views(pts_eval, obs, geom, min_view_m, max_view_m)
                if any(v is None for v in views.values()):
                    continue
                score = max(views.values()) + 0.1 * (a + b + c)
                if best is None or score < best[0]:
                    best = (score, steps, dict(clear=clear, views=views,
                                               legs=(a, b, c), obs=obs))
    return best


def run_step(argv: list) -> bool:
    print(f'>>> {" ".join(argv)}', flush=True)
    result = subprocess.run(argv)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Approach an obstacle on the front ultrasonic, then '
                    'drive one square loop around it (camera faces left).')
    parser.add_argument('--stop-cm', type=float, default=30.0,
                        help='stop the approach once the obstacle is this '
                             'close, cm, default 25')
    parser.add_argument('--leg-m', type=float, default=0.5,
                        help='length of each of the 4 sides of the loop, '
                             'metres, default 0.5 (must clear the obstacle '
                             'corner to corner)')
    parser.add_argument('--turn-deg', type=float, default=90.0,
                        help='turn angle at each corner, default 90')
    parser.add_argument('--direction', type=str, default='left',
                        choices=['left', 'right'],
                        help='turn direction for all 4 corners, default left '
                             '(keeps the obstacle on the left-facing camera '
                             'side for the whole loop)')
    parser.add_argument('--approach-speed', type=float, default=0.12,
                        help='m/s while approaching, default 0.12')
    parser.add_argument('--approach-timeout', type=float, default=20.0,
                        help='seconds to give up approaching if nothing is '
                             'seen, default 20')
    parser.add_argument('--drive-speed', type=float, default=0.15,
                        help='m/s for each straight leg, default 0.15')
    parser.add_argument('--turn-speed', type=float, default=0.15,
                        help='m/s while turning, default 0.15')
    parser.add_argument('--obstacle-cm', type=float, default=10.0,
                        help='obstacle cube side, cm, default 10')
    parser.add_argument('--front-cm', type=float, default=19.0,
                        help='rear axle -> front ultrasonic, cm (ESTIMATE - measure it)')
    parser.add_argument('--rear-cm', type=float, default=6.0,
                        help='rear axle -> back of car, cm (estimate)')
    parser.add_argument('--half-width-cm', type=float, default=8.0,
                        help='half the car width, cm (estimate)')
    parser.add_argument('--cam-fwd-cm', type=float, default=7.0,
                        help='rear axle -> camera, forward, cm (ESTIMATE - measure it)')
    parser.add_argument('--min-view-cm', type=float, default=10.0,
                        help='closest usable camera-to-face distance, cm')
    parser.add_argument('--max-view-cm', type=float, default=50.0,
                        help='farthest usable camera-to-face distance, cm')
    parser.add_argument('--margin-cm', type=float, default=6.0,
                        help='minimum body-to-obstacle clearance, cm')
    parser.add_argument('--plan-only', action='store_true',
                        help='print the planned path and exit without driving '
                             '(uses --stop-cm as the assumed stop distance)')
    parser.add_argument('--skip-approach', action='store_true',
                        help='skip the ultrasonic approach phase and start '
                             'the loop immediately (robot already in place)')
    args, ros_args = parser.parse_known_args()

    geom = dict(front=args.front_cm / 100.0, rear=args.rear_cm / 100.0,
                half_w=args.half_width_cm / 100.0, cam_fwd=args.cam_fwd_cm / 100.0)
    stop_range_m = args.stop_cm / 100.0

    if not (args.skip_approach or args.plan_only):
        rclpy.init(args=ros_args)
        try:
            reached, final_range = approach(args.stop_cm, args.approach_speed,
                                            args.approach_timeout)
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        if not reached:
            print('!!! approach did not reach the obstacle, aborting loop',
                  file=sys.stderr)
            return 1
        if math.isfinite(final_range):
            stop_range_m = final_range
        print(f'stopped {stop_range_m * 100:.1f} cm from the obstacle', flush=True)

    plan = plan_loop(args.direction, stop_range_m, args.obstacle_cm / 100.0, geom,
                     args.min_view_cm / 100.0, args.max_view_cm / 100.0,
                     args.margin_cm / 100.0)
    if plan is None:
        print('!!! no path clears the obstacle by '
              f'{args.margin_cm:.0f} cm AND shows all 4 faces to the left camera '
              f'from {args.min_view_cm:.0f}-{args.max_view_cm:.0f} cm, starting '
              f'{stop_range_m * 100:.0f} cm away. Try --stop-cm 30-40.', file=sys.stderr)
        return 1
    _, steps, info = plan
    print('PLAN (direction %s, stop %.0f cm):' % (args.direction, stop_range_m * 100))
    for kind, v in steps:
        print('   turn %s %.0f deg' % ('left' if v > 0 else 'right', abs(v))
              if kind == 'turn' else ('   leg %.2f m' % v))
    print('   body clearance to obstacle: %.0f cm' % (info['clear'] * 100))
    print('   camera-to-face distance: ' +
          ', '.join('%s %.0f cm' % (k, v * 100) for k, v in info['views'].items()),
          flush=True)
    if args.plan_only:
        return 0

    for n, (kind, v) in enumerate(steps, 1):
        if kind == 'turn':
            direction = 'left' if v > 0 else 'right'
            argv = ['ros2', 'run', 'mdp_bringup', 'rotate_angle.py',
                    str(abs(v)), '--direction', direction,
                    '--speed', str(args.turn_speed)]
        else:
            if v < 0.01:
                continue
            argv = ['ros2', 'run', 'mdp_bringup', 'drive_distance.py',
                    str(v), '--speed', str(args.drive_speed)]
        print(f'--- step {n}/{len(steps)} ---', flush=True)
        if not run_step(argv):
            print(f'!!! step {n} failed, aborting loop', file=sys.stderr)
            return 1

    print('>>> DONE - one full loop around the obstacle complete', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
