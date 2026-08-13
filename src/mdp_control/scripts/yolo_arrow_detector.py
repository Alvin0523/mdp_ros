#!/usr/bin/env python3
"""
Ultralytics YOLO ROS2 Arrow Detection Node.
Follows Ultralytics ROS & Raspberry Pi integration guides.
Subscribes to camera feed (/image_raw or /camera/image_raw),
runs YOLO inference, and publishes arrow direction ('LEFT' or 'RIGHT') on /arrow_detection.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class YoloArrowDetector(Node):
    def __init__(self):
        super().__init__('yolo_arrow_detector')
        
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('model_path', 'yolov8n.pt')
        
        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('model_path').value

        self.bridge = CvBridge()
        self.arrow_pub = self.create_publisher(String, '/arrow_detection', 10)
        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        if ULTRALYTICS_AVAILABLE:
            self.model = YOLO(model_path)
            self.get_logger().info(f"Ultralytics YOLO loaded successfully from {model_path}!")
        else:
            self.model = None
            self.get_logger().warn("Ultralytics library not installed. Running in simulation fallback mode!")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        if self.model is not None:
            # Run Ultralytics YOLO inference
            results = self.model(cv_image, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id].upper()
                    
                    if 'LEFT' in label or 'RIGHT' in label:
                        detected_direction = 'LEFT' if 'LEFT' in label else 'RIGHT'
                        self.publish_detection(detected_direction)
                        return
        else:
            # Fallback test simulation detection
            self.publish_detection('RIGHT')

    def publish_detection(self, direction: str):
        msg = String()
        msg.data = direction
        self.arrow_pub.publish(msg)
        self.get_logger().info(f"YOLO Arrow Detected: {direction}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloArrowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
