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

import rclpy
from rclpy.node import Node
from enum import Enum, auto
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from mdp_algorithm.collision_aware_planner import plan_route
from mdp_algorithm.occupancy_map import coords_to_grid
from mdp_algorithm.pure_pursuit_follower import PurePursuitController, yaw_from_quaternion


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

        # Publishers & Subscribers
        # /cmd_vel is the correct target: real.launch.py and sim.launch.py both
        # remap ackermann_steering_controller's actual reference subscription
        # to /cmd_vel, so publishing directly to
        # /ackermann_steering_controller/reference (as this used to do) never
        # reached the controller - dead traffic to an unsubscribed topic.
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bt_pub = self.create_publisher(String, '/bluetooth_tx', 10)

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
        self.leg_paths = []          # one dense path (metres) per obstacle in visiting_order
        self.unreachable = []
        self.current_target_idx = 0

        self.detected_target_id = None
        self.state_start_time = self.get_now_sec()
        self.get_logger().info("Task 1 Runner Node Initialized! Waiting for obstacle setup...")

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
            # /obstacle_setup gives obstacle position in METRES (matches
            # this node's own start_pose/current_pose convention) - convert
            # to the grid-cell coordinates collision_aware_planner expects.
            obstacles_grid = []
            for x_m, y_m, face in self.obstacles:
                x_g, y_g = coords_to_grid(x_m * 100.0, y_m * 100.0)
                obstacles_grid.append((x_g, y_g, face))

            # NOT (0.0, 0.0, ...): confirmed by direct testing that a rear-axle
            # start pose literally at the arena's inside corner is physically
            # unreachable for Hybrid A* to escape - the car's own front bumper
            # is already inside the 15cm wall-inflation margin on two sides at
            # once, and every turn choice's swept front point re-enters that
            # margin before the car can clear it (a real kinematic constraint,
            # not a search bug - a car with this turning radius genuinely
            # cannot escape a corner it's touching without violating its own
            # safety margin mid-turn). (0.15, 0.15) - one inflation-margin-width
            # in from both walls - is a reasonable placeholder for "rear axle at
            # the inner corner of the marked start box," but is NOT verified
            # against the actual competition start-box convention/size. Confirm
            # against the real starting setup before trusting this for a run.
            start_pose = (0.15, 0.15, math.pi / 2)
            self.visiting_order, self.leg_paths, self.unreachable = plan_route(obstacles_grid, start_pose)

            if self.unreachable:
                self.get_logger().warn(f"Obstacles with no valid scan checkpoint, skipped: {self.unreachable}")
            self.get_logger().info(f"Visiting Order Calculated: {self.visiting_order}")

            self.current_target_idx = 0
            self.state = State.NAVIGATING_TO_TARGET
            self.state_start_time = now
            self._start_current_leg()

        # STATE 2: Navigating to current target standoff pose - now actually
        # tracks the Hybrid A*-planned path via PurePursuitController,
        # instead of blindly driving forward for a fixed 2.5s.
        elif self.state == State.NAVIGATING_TO_TARGET:
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
                    self._start_current_leg()

        # STATE 4: Finished & Auto-Stopped
        elif self.state == State.FINISHED:
            self.send_cmd(0.0, 0.0)

    def _start_current_leg(self):
        """Load the current target's pre-planned path into the follower.
        An empty leg (Hybrid A* found no path - see collision_aware_planner
        docstring) is treated as already-done so the state machine skips
        straight to the scan pause rather than getting stuck driving
        nowhere."""
        path = self.leg_paths[self.current_target_idx]
        if not path:
            self.get_logger().warn(f"No path found to target {self.current_target_idx + 1}, skipping navigation.")
        self.follower.set_path(path)


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
