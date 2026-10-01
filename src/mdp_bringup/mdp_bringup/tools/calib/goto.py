"""`calib goto`: drive to one tablet cell with the task 1 planner and follower
(one leg) - checks them end to end: does the car stop in the right cell,
facing the right way?

    pixi run calib goto 5 8 E                  rear axle (base_link) to cell (5,8), facing E
    pixi run calib goto 5 8 E --no-obstacles   plan in the empty arena (walls only)
    pixi run calib goto 5 8 E --layout my.yaml blocks from another layout file
    pixi run calib goto 5 8 E --speed 0.2      slower than the task 1 speed

Cells as on the tablet: col, row 0..19, (0,0) bottom-left; the car stops with
the centre of its rear axle over the centre of that cell. The blocks it plans
around are task1 of config/tasks.yaml (what the sim arena has) unless told
otherwise. Settings (speed, lookahead, tolerances): config/navigation.yaml.

For the BARE car (task:=0): `calib` refuses to start next to a task runner,
and this stops if one appears while driving. Ctrl+C stops the car.

THE LOG: on arrival it asks how far the middle of the rear axle is from the
cell centre (in sim it takes Gazebo's true pose) and adds a row to
calibration_log.csv (tools/calib/log.py): the car's own miss vs the true one.
"""
import argparse
import math

import tf2_geometry_msgs  # noqa: F401  registers the geometry_msgs conversions tf2 uses
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from mdp_algorithm.control.pure_pursuit_follower import PurePursuitController, yaw_from_quaternion
from mdp_algorithm.planning.costmap import Costmap, Obstacle
from mdp_algorithm.planning.planner import plan_leg
from mdp_bringup.tools.calib import log
from mdp_bringup.utils import markers, obstacle_layout
from mdp_bringup.utils.run import run, wall_timer

HEADING = {'E': 0.0, 'N': math.pi / 2.0, 'W': math.pi, 'S': -math.pi / 2.0}
RUNNERS = ('task1_runner', 'task2_runner')


def cell_centre(c: int) -> float:
    return (c + 0.5) * markers.CELL_M


class GoTo(Node):
    def __init__(self, target, blocks, speed=None):
        super().__init__('goto')
        self.speed = speed                         # m/s, None = navigation.yaml's desired_linear_vel
        self.target = target                       # (x, y, yaw) metres, map
        self.blocks = blocks                       # obstacle_layout.LayoutObstacle
        self.pose = None
        self.follower = None
        self.truth = log.Truth(self)
        self.start_time = None
        self.state = 'WAIT_POSE'
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 10)
        wall_timer(self, 0.05, self.tick)          # 20 Hz, like task1_runner

    def fmt(self, pose) -> str:
        return f"({markers.cell(pose[0])},{markers.cell(pose[1])}){markers.direction(pose[2])}"

    def on_odom(self, msg: Odometry):
        """EKF pose (odom) -> map, where the cells are."""
        try:
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException:
            return
        p = tf2_geometry_msgs.do_transform_pose(msg.pose.pose, tf)
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        if self.follower is not None:
            self.follower.update_pose(*self.pose)

    def send(self, v: float, w: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def runner_running(self) -> bool:
        busy = [n for n in RUNNERS if n in self.get_node_names()]
        if busy:
            self.get_logger().error(f"{busy[0]} is running and owns /cmd_vel - "
                                    f"start the bare car (task:=0) to use goto")
        return bool(busy)

    def off_path(self) -> bool:
        """Off the planned path (task1_runner.left_path's rule): armed once on it,
        and past twice the limit regardless."""
        err, limit = self.follower.path_error(), self.follower.max_path_error
        if err <= limit / 2.0:
            self._on_path = True
        return err > (limit if getattr(self, '_on_path', False) else 2.0 * limit)

    def tick(self):
        if self.state == 'WAIT_POSE':
            if self.pose is not None:
                self.plan()
        elif self.state == 'DRIVE':
            if self.runner_running():   # started while driving
                self.send(0.0, 0.0)
                self.finish()
                return
            cmd = self.follower.compute_cmd()
            if cmd is None or self.follower.is_done():
                self.send(0.0, 0.0)
                yaw_err = math.degrees(math.atan2(math.sin(self.pose[2] - self.target[2]),
                                                  math.cos(self.pose[2] - self.target[2])))
                self.get_logger().info(f"ARRIVED   at {self.fmt(self.pose)} (target {self.fmt(self.target)}, "
                                       f"heading {yaw_err:+.0f}deg)")
                self.write_log(yaw_err)
                self.finish()
            elif self.off_path():
                # Off the planned path = where the planner never checked for blocks.
                self.send(0.0, 0.0)
                self.get_logger().error(f"OFF PATH  {self.follower.path_error() * 100:.0f} cm off at "
                                        f"{self.fmt(self.pose)} - stopped (limit "
                                        f"{self.follower.max_path_error * 100:.0f} cm)")
                self.finish()
            else:
                self.send(*cmd)

    def plan(self):
        self.state = 'PLAN'
        self.get_logger().info(f"PLAN      {self.fmt(self.pose)} -> {self.fmt(self.target)} "
                               f"around {len(self.blocks)} block(s)")
        costmap = Costmap([Obstacle(o.x * 100.0, o.y * 100.0, o.facing, i) for i, o in enumerate(self.blocks)])
        if costmap.in_collision(self.target[0] * 100.0, self.target[1] * 100.0, self.target[2]):
            self.get_logger().error(f"{self.fmt(self.target)}: the car does not fit there (a block, the table "
                                    f"edge, or within the safety margin of one)")
            self.finish()
            return
        if costmap.in_collision(self.pose[0] * 100.0, self.pose[1] * 100.0, self.pose[2]):
            self.get_logger().error(f"the car at {self.fmt(self.pose)} is off the table, or closer to a block "
                                    f"than the safety margin (navigation.yaml footprint_padding) - "
                                    f"move it clear (and `pixi run reset`)")
            self.finish()
            return
        path = plan_leg(costmap, self.pose, self.target)
        if not path:
            self.get_logger().error(f"NO PATH   to {self.fmt(self.target)}")
            self.finish()
            return
        self.path_pub.publish(markers.route_path([path], self.get_clock().now().to_msg()))
        gears = [p[3] for p in path]
        moves = ' '.join('fwd' if gears[i] >= 0 else 'REV'
                         for i in range(len(gears)) if i == 0 or gears[i] != gears[i - 1])
        self.get_logger().info(f"GO        {moves}")
        self.follower = PurePursuitController(target_speed=self.speed)
        self.follower.update_pose(*self.pose)
        self.follower.set_path(path)
        self.state = 'DRIVE'
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def write_log(self, yaw_err):
        """The run's row in calibration_log.csv: how far off the cell centre, car vs true."""
        where = log.where(self)
        tx, ty, tyaw = self.target
        car = math.hypot(self.pose[0] - tx, self.pose[1] - ty) * 100.0
        true_yaw = None
        if where == 'sim' and self.truth.pose is not None:
            x, y, yaw = self.truth.pose
            true = math.hypot(x - tx, y - ty) * 100.0
            true_yaw = math.degrees(math.atan2(math.sin(yaw - tyaw), math.cos(yaw - tyaw)))
        else:
            true = log.ask('TAPE      middle of the rear axle -> cell centre (cm)')
        secs = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        notes = f'heading off {yaw_err:+.0f} deg (car)'
        if true_yaw is not None:
            notes += f' / {true_yaw:+.0f} deg (true)'
        notes += f'; {secs:.1f} s'
        log.write(where, 'goto', self.fmt(self.target), self.follower.target_speed, car, true, 'cm off', notes)

    def finish(self):
        self.state = 'DONE'
        raise SystemExit

    def destroy_node(self):
        try:
            self.send(0.0, 0.0)   # never leave the car driving (Ctrl+C mid-leg)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    ap = argparse.ArgumentParser(description='Drive to a tablet cell: calib goto COL ROW N|E|S|W')
    ap.add_argument('col', type=int)
    ap.add_argument('row', type=int)
    ap.add_argument('dir', type=str.upper, choices=list(HEADING))
    ap.add_argument('--no-obstacles', action='store_true', help='plan in the empty arena')
    ap.add_argument('--speed', type=float, default=None,
                    help='m/s, clamped 0.05 - 0.5 (default: navigation.yaml follower.desired_linear_vel, the task 1 speed)')
    ap.add_argument('--layout', default=f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml",
                    help='layout file whose task1 blocks to avoid (default: config/tasks.yaml)')
    a, ros_args = ap.parse_known_args()
    if not (0 <= a.col <= 19 and 0 <= a.row <= 19):
        ap.error('cells are 0..19')
    target = (cell_centre(a.col), cell_centre(a.row), HEADING[a.dir])
    speed = None if a.speed is None else min(0.5, max(0.05, a.speed))
    blocks = [] if a.no_obstacles else obstacle_layout.load(a.layout, 'task1')
    try:
        run(lambda: GoTo(target, blocks, speed), args=ros_args)
    except SystemExit:
        pass


if __name__ == '__main__':
    main()
