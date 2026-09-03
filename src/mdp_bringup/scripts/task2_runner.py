#!/usr/bin/env python3
"""
True Odometry & Ground Truth Position-Reaching Task 2 Runner Node.
Features adaptive cornering speed reduction (reduces linear speed on sharp turns
to allow full HWZ020 servo yaw rotation without overshooting).
"""

import math
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from enum import Enum, auto
from geometry_msgs.msg import TwistStamped, PoseStamped, Point
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import tf2_ros

from mdp_algorithm.spline_planner import SplinePathPlanner

class State(Enum):
    WAITING_FOR_START = auto()
    DRIVING_PATH = auto()
    FINISHED = auto()

class Task2Runner(Node):
    def __init__(self):
        super().__init__('task2_runner')
        
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        
        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        
        self.create_subscription(String, '/yolo_result', self.arrow_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        self.create_subscription(Odometry, '/ackermann_steering_controller/odometry', self.odom_callback, 10)
        
        # TF Buffer & Listener for Ground Truth Gazebo Pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Control Loop @ 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Ackermann Kinematics Specs (must match ackermann_controller.yaml)
        self.wheelbase = 0.1433      # L = 0.1433 m
        self.wheel_radius = 0.0325   # R = 0.0325 m
        self.max_steering = 0.39     # HWZ020 limit (22.35 deg)
        self.lookahead = 0.30        # Pure Pursuit Lookahead distance (30cm)
        self.min_turn_radius = self.wheelbase / math.tan(self.max_steering)
        self.kappa_max = math.tan(self.max_steering) / self.wheelbase  # max path curvature (1/R_min)

        # Spline path planner: smooth (x,y,yaw) curve through the waypoints, with
        # no manual heading-guessing at interior points (that's what made the
        # earlier hand-rolled Dubins chaining loop wildly). Curvature is clamped
        # to kappa_max by the pure-pursuit controller at tracking time.
        self.planner = SplinePathPlanner()
        self.dense_path: List[Tuple[float, float, float]] = []
        self.path_idx = 0

        # Static arena geometry (mirrors task2_arena.sdf) so it can be visualized in
        # Foxglove too, since Foxglove never sees the Gazebo-only SDF world file.
        # name, (x, y, z), (size_x, size_y, size_z), (r, g, b, a)
        self.ARENA_GEOMETRY = [
            ("obstacle_1", (1.0, 0.0, 0.10), (0.10, 0.10, 0.20), (0.8, 0.2, 0.2, 0.9)),
            ("obstacle_2", (1.8, 0.0, 0.10), (0.10, 0.30, 0.20), (0.2, 0.2, 0.8, 0.9)),
            ("carpark_zone", (0.0, 0.0, 0.001), (0.6, 0.5, 0.002), (0.2, 0.8, 0.2, 0.4)),
        ]

        # Configurable Competition Distances
        self.obs1_x = 1.0
        self.obs2_x = 1.8
        self.sweep_offset = 0.50     # Lateral clearance offset (50 cm)

        # Robot Positions
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_yaw = 0.0

        # State & Decisions
        self.state = State.WAITING_FOR_START
        self.arrow1 = 'LEFT'
        self.arrow2 = 'RIGHT'
        
        self.waypoints: List[Tuple[float, float]] = []
        self.current_wpt_idx = 0
        self.state_start_time = self.get_now_sec()
        self.get_logger().info("Task 2 Adaptive Cornering Runner Initialized! Waiting 6s for bringup...")

    def get_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def arrow_callback(self, msg):
        direction = msg.data.upper()
        if self.state == State.WAITING_FOR_START or self.current_wpt_idx < 3:
            self.arrow1 = direction
            self.get_logger().info(f"Arrow 1 Set to: {self.arrow1}")
        else:
            self.arrow2 = direction
            self.get_logger().info(f"Arrow 2 Set to: {self.arrow2}")

    def odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_pub.publish(pose_msg)

    def update_ground_truth_tf(self):
        try:
            t = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
            self.gt_x = t.transform.translation.x
            self.gt_y = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.gt_yaw = math.atan2(siny_cosp, cosy_cosp)
        except Exception:
            self.gt_x = self.odom_x
            self.gt_y = self.odom_y
            self.gt_yaw = self.odom_yaw

    def joint_states_callback(self, msg: JointState):
        pass

    def generate_task2_slalom_waypoints(self) -> List[Tuple[float, float]]:
        side1 = 1.0 if self.arrow1 == 'LEFT' else -1.0
        side2 = -1.0 if self.arrow1 == 'LEFT' else 1.0

        wpts = [
            (0.35, 0.0),                                       # Wpt 0: Carpark Exit Sprint
            (1.15, 0.50 * side1),                              # Wpt 1: Clear of Obstacle 1, same side (0.10m buffer past obs1 edge at x=1.05)
            (1.40, 0.00),                                      # Wpt 2: Crossover midpoint (centerline, between obstacles)
            (1.65, 0.50 * side2),                              # Wpt 3: Opposite side, clear before Obstacle 2 (0.10m buffer before obs2 edge at x=1.75)
            (1.95, 0.50 * side2),                              # Wpt 4: Clear past Obstacle 2 (0.10m buffer past obs2 edge at x=1.85)
            (2.20, 0.00),                                      # Wpt 5: Loop entry behind Obstacle 2
            (1.95, 0.50 * (-side2)),                           # Wpt 6: Pass other side of Obstacle 2 (mirrors Wpt 4 buffer)
            (1.00, 0.20 * (-side2)),                           # Wpt 7: Curve onto return axis
            (0.00, 0.00)                                       # Wpt 8: Straight back into Carpark & Stop
        ]
        return wpts

    def publish_visualizations(self):
        if not self.waypoints:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        # Publish the actual dense, curvature-feasible path being followed
        # (falls back to the sparse waypoints if planning hasn't run yet)
        source = self.dense_path if self.dense_path else [(x, y, 0.0) for x, y in self.waypoints]
        for x, y, _ in source:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = 0.05
            path_msg.poses.append(p)
        self.path_pub.publish(path_msg)

        marker_array = MarkerArray()

        # Thick line strip - easier to spot than the raw Path topic in Foxglove
        line = Marker()
        line.header = path_msg.header
        line.ns = "dense_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.02
        line.color.r = 0.1
        line.color.g = 0.6
        line.color.b = 1.0
        line.color.a = 0.9
        for x, y, _ in source:
            pt = Point()
            pt.x, pt.y, pt.z = float(x), float(y), 0.03
            line.points.append(pt)
        marker_array.markers.append(line)

        # Static arena geometry (mirrors task2_arena.sdf) - Foxglove never sees the
        # Gazebo-only SDF world, so mirror the obstacle/carpark boxes here.
        for i, (name, (ox, oy, oz), (sx, sy, sz), color) in enumerate(self.ARENA_GEOMETRY):
            box = Marker()
            box.header = path_msg.header
            box.ns = "arena"
            box.id = i
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(ox)
            box.pose.position.y = float(oy)
            box.pose.position.z = float(oz)
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = float(sx), float(sy), float(sz)
            box.color.r, box.color.g, box.color.b, box.color.a = color
            marker_array.markers.append(box)

        for idx, (x, y) in enumerate(self.waypoints):
            sphere = Marker()
            sphere.header = path_msg.header
            sphere.ns = "waypoints"
            sphere.id = idx
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(x)
            sphere.pose.position.y = float(y)
            sphere.pose.position.z = 0.05
            sphere.scale.x = 0.12
            sphere.scale.y = 0.12
            sphere.scale.z = 0.12
            
            if idx == self.current_wpt_idx:
                sphere.color.r = 0.0
                sphere.color.g = 1.0
                sphere.color.b = 0.0
            else:
                sphere.color.r = 1.0
                sphere.color.g = 1.0
                sphere.color.b = 0.0
            sphere.color.a = 0.9
            marker_array.markers.append(sphere)

            text = Marker()
            text.header = path_msg.header
            text.ns = "waypoint_labels"
            text.id = 100 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = 0.20
            text.scale.z = 0.10
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"W{idx} ({x:.2f}, {y:.2f})"
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def send_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def find_lookahead_point(self) -> int:
        """Advances self.path_idx monotonically along the dense path (never chases
        backwards) and returns the index of the first point at/beyond lookahead
        distance from the current ground-truth position."""
        n = len(self.dense_path)
        while self.path_idx < n - 1:
            px, py, _ = self.dense_path[self.path_idx]
            if math.hypot(px - self.gt_x, py - self.gt_y) < self.lookahead * 0.5:
                self.path_idx += 1
            else:
                break

        for i in range(self.path_idx, n):
            px, py, _ = self.dense_path[i]
            if math.hypot(px - self.gt_x, py - self.gt_y) >= self.lookahead:
                return i
        return n - 1

    def control_loop(self):
        now = self.get_now_sec()
        elapsed = now - self.state_start_time

        self.update_ground_truth_tf()

        # STATE 0: Waiting for bringup (6s)
        if self.state == State.WAITING_FOR_START:
            self.send_cmd(0.0, 0.0)
            if elapsed > 6.0:
                self.waypoints = self.generate_task2_slalom_waypoints()
                # Plan a single curvature-feasible path through all waypoints up front
                # (Dubins, constrained to self.min_turn_radius) instead of beelining
                # at each raw waypoint - this is what actually respects the car's
                # Ackermann turning-radius limit and avoids corner-cutting.
                self.dense_path = self.planner.generate_path_through_waypoints(
                    self.waypoints, step_size=0.05
                )
                peak_kappa = self.planner.max_curvature(self.dense_path)
                self.path_idx = 0
                self.current_wpt_idx = 0
                self.state = State.DRIVING_PATH
                self.state_start_time = now
                feasible = "OK" if peak_kappa <= self.kappa_max else "EXCEEDS kappa_max!"
                self.get_logger().info(
                    f"Starting Task 2 Execution! Planned {len(self.dense_path)} dense path points | "
                    f"peak curvature={peak_kappa:.2f} (1/m), kappa_max={self.kappa_max:.2f} [{feasible}]"
                )

        # STATE 1: Driving Path (pure pursuit over the planned Dubins path)
        elif self.state == State.DRIVING_PATH:
            self.publish_visualizations()

            final_x, final_y, _ = self.dense_path[-1]
            dist_to_end = math.hypot(final_x - self.gt_x, final_y - self.gt_y)

            if self.path_idx >= len(self.dense_path) - 1 and dist_to_end < 0.10:
                self.send_cmd(0.0, 0.0)
                self.state = State.FINISHED
                self.get_logger().info("Task 2 Slalom Path Completed! Stopped in Carpark.")
                return

            target_idx = self.find_lookahead_point()
            target_x, target_y, _ = self.dense_path[target_idx]

            dx = target_x - self.gt_x
            dy = target_y - self.gt_y
            local_x = dx * math.cos(-self.gt_yaw) - dy * math.sin(-self.gt_yaw)
            local_y = dx * math.sin(-self.gt_yaw) + dy * math.cos(-self.gt_yaw)

            lookahead_actual = max(math.hypot(local_x, local_y), 0.05)
            curvature = (2.0 * local_y) / (lookahead_actual ** 2)
            # Clamp to what the vehicle can actually steer - the planned path never
            # exceeds this, but the pursuit geometry near sharp junctions still can.
            curvature = max(-self.kappa_max, min(self.kappa_max, curvature))

            # ADAPTIVE SPEED: slow down when tracking near the steering limit
            if abs(curvature) > 0.5 * self.kappa_max:
                linear_speed = 0.22
            else:
                linear_speed = 0.45

            angular_speed = linear_speed * curvature
            self.send_cmd(linear_x=linear_speed, angular_z=angular_speed)

            # Milestone logging only (does not drive control anymore)
            if self.current_wpt_idx < len(self.waypoints):
                wx, wy = self.waypoints[self.current_wpt_idx]
                if math.hypot(wx - self.gt_x, wy - self.gt_y) < 0.15:
                    odom_err = math.hypot(wx - self.odom_x, wy - self.odom_y)
                    self.get_logger().info(
                        f"[Wpt {self.current_wpt_idx}] Target({wx:.2f}, {wy:.2f}) | "
                        f"Gazebo GT({self.gt_x:.2f}, {self.gt_y:.2f}) | "
                        f"Odom ({self.odom_x:.2f}, {self.odom_y:.2f}) [Err: {odom_err:.2f}m]"
                    )
                    self.current_wpt_idx += 1

        # STATE 2: Finished
        elif self.state == State.FINISHED:
            self.publish_visualizations()
            self.send_cmd(0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    node = Task2Runner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
