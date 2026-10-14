#!/usr/bin/env python3
"""Bullseye boxes for task 2 without a YOLO class for them (bullseye.py: template
matching), on /bullseye_detections in /yolo_detections' format:

    {"stamp": s, "w": 640, "h": 480, "boxes": [["99", score, x1, y1, x2, y2], ...]}

task2_runner reads both topics; once YOLO reports bullseyes itself (class 99),
start the launch with bullseye_matcher:=false. Runs where YOLO runs (the laptop
with `pixi run laptop`; with role:=solo for task 2). Settings: config/vision.yaml.
"""

import json
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from mdp_vision.bullseye import find_bullseyes, load_template

BULLSEYE_ID = '99'


class BullseyeMatcher(Node):
    def __init__(self):
        super().__init__('bullseye_matcher')
        camera_topic = self.declare_parameter('camera_topic', '/image_raw').value
        self.min_px = float(self.declare_parameter('min_px', 14.0).value)
        self.max_px = float(self.declare_parameter('max_px', 140.0).value)
        self.threshold = float(self.declare_parameter('threshold', 0.6).value)
        template = self.declare_parameter('template', os.path.join(
            get_package_share_directory('mdp_description'), 'models', 'symbols', '99_Bullseye.png')).value
        self.template = load_template(template)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(String, '/bullseye_detections', 10)
        # The newest frame only (as YOLO): matching takes longer than a frame.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.compressed = camera_topic.endswith('/compressed')
        self.create_subscription(CompressedImage if self.compressed else Image, camera_topic, self.on_image, qos)
        self.get_logger().info(f"BULLSEYE  template matching on {camera_topic}")

    def on_image(self, msg):
        try:
            if self.compressed:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='mono8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as exc:
            self.get_logger().error(f"BULLSEYE  bad image: {exc}")
            return
        found = find_bullseyes(frame, self.template, self.min_px, self.max_px, self.threshold)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        boxes = [[BULLSEYE_ID, round(score, 3)] + [round(v, 1) for v in box] for score, *box in found]
        self.pub.publish(String(data=json.dumps({'stamp': stamp, 'w': frame.shape[1], 'h': frame.shape[0],
                                                 'boxes': boxes})))


def main(args=None):
    rclpy.init(args=args)
    node = BullseyeMatcher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (Exception, KeyboardInterrupt):
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
