#!/usr/bin/env python3
"""
Ultralytics YOLO ROS2 Arrow & Obstacle Detector Node.
Located in mdp_vision package.
Subscribes to camera feed (/image_raw), runs YOLO inference,
and publishes detected target/arrow string to /yolo_result.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

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
        self.declare_parameter('result_topic', '/yolo_result')
        
        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('model_path').value
        result_topic = self.get_parameter('result_topic').value

        self.bridge = CvBridge()
        self.result_pub = self.create_publisher(String, result_topic, 10)
        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        if ULTRALYTICS_AVAILABLE:
            self.model = YOLO(model_path)
            self.get_logger().info(f"[mdp_vision] Ultralytics YOLO loaded successfully from {model_path}!")
        else:
            self.model = None
            self.get_logger().warn("[mdp_vision] Ultralytics library not installed. Simulation fallback mode active.")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        if self.model is not None:
            results = self.model(cv_image, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id].upper()
                    self.publish_detection(label)
                    return

    def publish_detection(self, detection: str):
        msg = String()
        msg.data = detection
        self.result_pub.publish(msg)
        self.get_logger().info(f"[mdp_vision] YOLO Detected: {detection}")

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
