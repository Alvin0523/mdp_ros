#!/usr/bin/env python3
"""
Dedicated Camera Publisher Node for MDP Vision.
Captures frames from Raspberry Pi / USB Camera (/dev/video0)
and publishes ROS 2 Image messages to /image_raw.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('video_device', 0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('camera_topic', '/image_raw')

        video_device = self.get_parameter('video_device').value
        self.width = self.get_parameter('image_width').value
        self.height = self.get_parameter('image_height').value
        frame_rate = self.get_parameter('frame_rate').value
        camera_topic = self.get_parameter('camera_topic').value

        self.publisher = self.create_publisher(Image, camera_topic, 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(video_device)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.get_logger().info(f"Camera opened on /dev/video{video_device} ({self.width}x{self.height} @ {frame_rate} FPS)")
        else:
            self.get_logger().warn(f"Failed to open camera on /dev/video{video_device}. Will publish fallback frames.")

        timer_period = 1.0 / frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_frame'
                self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
