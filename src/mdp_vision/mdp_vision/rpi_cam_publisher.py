#!/usr/bin/env python3
"""
RPi Camera Publisher Node for MDP Vision.

Captures raw YUV420 frames from the Pi Camera Module (IMX219) via
`rpicam-vid` (no libcamera/camera_ros build required) and republishes them as
ROS 2 sensor_msgs/Image (bgr8) on `camera_topic` - same topic/type
yolo_detector.py already expects.

Single-threaded, fixed-size reads: `rpicam-vid --codec yuv420` emits frames
of exactly width*height*3/2 bytes each with no framing/parsing needed (unlike
MJPEG, which requires scanning for SOI/EOI markers in a byte stream). This
was benchmarked standalone at a steady 30.0-30.1 fps @ 640x480 on this same
Pi 4B (no drops over a sustained run) - see docs/pi-camera-vision.md.

Also publishes a JPEG-compressed copy on `<camera_topic>/compressed`, but
only while something is actually subscribed to it (e.g. Foxglove open for
live monitoring) - raw uncompressed frames are fine for the local YOLO
subscriber on the same host, but saturate a remote Foxglove websocket
client's decode at ~27MB/s, causing a growing playback lag; JPEG cuts that
to ~15-20KB/frame.
"""

import subprocess
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


class RpiCamPublisher(Node):
    def __init__(self):
        super().__init__('rpi_cam_publisher')

        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('jpeg_quality', 80)

        self.width = self.get_parameter('image_width').value
        self.height = self.get_parameter('image_height').value
        frame_rate = self.get_parameter('frame_rate').value
        camera_topic = self.get_parameter('camera_topic').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        self.frame_size = self.width * self.height * 3 // 2  # I420 (YUV420 planar)
        self.bridge = CvBridge()

        # BEST_EFFORT + depth 1 (KEEP_LAST): matches yolo_detector.py's
        # subscription QoS - if a consumer falls slightly behind, it always
        # gets handed the newest frame instead of working through a growing
        # backlog of stale ones. camera_ros's default publisher QoS is
        # RELIABLE, which a BEST_EFFORT subscriber is compatible with too.
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(Image, camera_topic, image_qos)
        self.compressed_pub = self.create_publisher(
            CompressedImage, f'{camera_topic}/compressed', image_qos)

        cmd = [
            'rpicam-vid',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(frame_rate),
            '--codec', 'yuv420',
            '--timeout', '0',
            '--nopreview',
            '--flush',
            '-o', '-',
        ]
        self.get_logger().info(
            f"rpicam-vid started ({self.width}x{self.height} @ {frame_rate} FPS), "
            f"publishing to {camera_topic}")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.frame_size)

        self.frame_count = 0
        self.window_count = 0
        self.start_time = time.monotonic()
        self.window_start = self.start_time
        self.report_period_s = 5.0

        self.timer = self.create_timer(0.0, self.tick)

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def tick(self):
        raw = self._read_exact(self.frame_size)
        if raw is None:
            self.get_logger().error('rpicam-vid stream ended, shutting down')
            self.timer.cancel()
            raise SystemExit

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        stamp = self.get_clock().now().to_msg()

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_frame'
        self.pub.publish(msg)

        # Skip JPEG encode entirely when nobody's watching - costs real CPU
        # that would otherwise go to capture/inference.
        if self.compressed_pub.get_subscription_count() > 0:
            ok, jpeg = cv2.imencode(
                '.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.header.frame_id = 'camera_frame'
                cmsg.format = 'jpeg'
                cmsg.data = jpeg.tobytes()
                self.compressed_pub.publish(cmsg)

        self.frame_count += 1
        self.window_count += 1

        elapsed = time.monotonic() - self.window_start
        if elapsed >= self.report_period_s:
            fps = self.window_count / elapsed
            avg_fps = self.frame_count / (time.monotonic() - self.start_time)
            self.get_logger().info(
                f'fps(window)={fps:.2f}  fps(avg)={avg_fps:.2f}  frames={self.frame_count}')
            self.window_count = 0
            self.window_start = time.monotonic()

    def destroy_node(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RpiCamPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
