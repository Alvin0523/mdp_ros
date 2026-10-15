#!/usr/bin/env python3
"""The Pi camera's JPEG stream, taken over the link once and republished on the
laptop (`pixi run laptop`, real car only) for every reader there.

A DDS sends each remote subscriber its own copy: yolo_detector and
bullseye_matcher reading /image_raw/compressed from the laptop took ~1.2 MB/s
each through the Tailscale relay on NTUSECURE, which carries ~1.7 MB/s - every
topic queued behind the frames, 0.5 s late (2026-10-15). So the readers there
take /laptop/image_raw/compressed instead and only this node reads the Pi's.
Frames pass as bytes (nothing decoded); the newest one only, like the readers.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class CameraRelay(Node):
    def __init__(self):
        super().__init__('camera_relay')
        source = self.declare_parameter('source', '/image_raw/compressed').value
        target = self.declare_parameter('target', '/laptop/image_raw/compressed').value
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(CompressedImage, target, qos)
        self.sub = None
        self.subscribe = lambda: self.create_subscription(CompressedImage, source, self.pub.publish, qos, raw=True)
        # Subscribed only while something here reads the copy: rpi_cam_publisher
        # stops sending the JPEGs when nobody subscribes.
        self.create_timer(1.0, self.follow_readers)
        self.get_logger().info(f"RELAY     {source} -> {target} (while read)")

    def follow_readers(self):
        wanted = self.pub.get_subscription_count() > 0
        if wanted and self.sub is None:
            self.sub = self.subscribe()
        elif not wanted and self.sub is not None:
            self.destroy_subscription(self.sub)
            self.sub = None


def main(args=None):
    rclpy.init(args=args)
    node = CameraRelay()
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
