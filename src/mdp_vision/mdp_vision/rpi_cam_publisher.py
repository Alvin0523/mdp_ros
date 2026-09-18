#!/usr/bin/env python3
"""
RPi Camera Publisher Node for MDP Vision.
Captures MJPEG frames from the Pi Camera Module (IMX219) via `rpicam-vid`
(hardware-accelerated, no libcamera/camera_ros build required) and republishes
them as ROS 2 sensor_msgs/Image, decoded, so this is a drop-in replacement for
camera_ros's camera_node - same topic/type yolo_detector.py already expects.

TEMP INSTRUMENTATION: logs parsed/decoded/published frame counts every second
to localize a throughput bottleneck (see get_logger().info in _log_stats).
"""

import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RpiCamPublisher(Node):
    def __init__(self):
        super().__init__('rpi_cam_publisher')

        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('camera_topic', '/camera/image_raw')

        self.width = self.get_parameter('image_width').value
        self.height = self.get_parameter('image_height').value
        frame_rate = self.get_parameter('frame_rate').value
        camera_topic = self.get_parameter('camera_topic').value

        self.publisher = self.create_publisher(Image, camera_topic, 10)
        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._latest_frame = None

        # Instrumentation counters, reset every second by _log_stats.
        self._stat_lock = threading.Lock()
        self._n_parsed = 0
        self._n_decoded = 0
        self._n_published = 0
        self._n_bytes_read = 0

        cmd = [
            'rpicam-vid',
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(frame_rate),
            '--codec', 'mjpeg',
            '-n',
            '--inline',
            '-o', '-',
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=65536)
        self.get_logger().info(
            f"rpicam-vid started ({self.width}x{self.height} @ {frame_rate} FPS), "
            f"publishing to {camera_topic}")

        self._stop = False
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self.timer = self.create_timer(1.0 / frame_rate, self.publish_latest)
        self.stats_timer = self.create_timer(1.0, self._log_stats)

    def _read_loop(self):
        buffer = bytearray()
        while not self._stop:
            chunk = self.proc.stdout.read(65536)
            if not chunk:
                break
            with self._stat_lock:
                self._n_bytes_read += len(chunk)
            buffer.extend(chunk)

            soi = buffer.find(b'\xff\xd8')
            eoi = buffer.find(b'\xff\xd9')
            if soi == -1 or eoi == -1 or eoi <= soi:
                continue

            jpeg_data = bytes(buffer[soi:eoi + 2])
            del buffer[:eoi + 2]
            with self._stat_lock:
                self._n_parsed += 1

            t0 = time.monotonic()
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            decode_ms = (time.monotonic() - t0) * 1000.0
            if frame is None:
                continue
            with self._stat_lock:
                self._n_decoded += 1
                self._last_decode_ms = decode_ms

            with self._lock:
                self._latest_frame = frame

    def publish_latest(self):
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        if frame is None:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.publisher.publish(msg)
        with self._stat_lock:
            self._n_published += 1

    def _log_stats(self):
        with self._stat_lock:
            parsed, decoded, published = self._n_parsed, self._n_decoded, self._n_published
            kb_read = self._n_bytes_read / 1024.0
            last_decode_ms = getattr(self, '_last_decode_ms', -1)
            self._n_parsed = self._n_decoded = self._n_published = self._n_bytes_read = 0
        self.get_logger().info(
            f"[stats/s] pipe_read={kb_read:.0f}KB parsed={parsed} decoded={decoded} "
            f"published={published} last_decode={last_decode_ms:.2f}ms")

    def destroy_node(self):
        self._stop = True
        if self.proc:
            self.proc.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RpiCamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
