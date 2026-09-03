#!/usr/bin/env python3
"""
Ultralytics YOLO ROS2 Detector Node.
Located in mdp_yolo package (mdp_vision/mdp_yolo).
Subscribes to camera feed (/image_raw), runs YOLO inference,
and publishes detected target/arrow string to /yolo_result.
"""

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# The `_ncnn_model` suffix is required, not a naming choice - ultralytics'
# AutoBackend detects the model format from the directory name itself
# (every export format has its own required suffix: *_ncnn_model/,
# *_saved_model/, *_openvino_model/, ...), not from the files inside it.
DEFAULT_MODEL_PATH = os.path.join(
    get_package_share_directory('mdp_yolo'), 'models', 'yolo26n_ncnn_model')

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('result_topic', '/yolo_result')
        self.declare_parameter('annotated_topic', '/yolo_result/image_annotated')

        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('model_path').value
        result_topic = self.get_parameter('result_topic').value
        annotated_topic = self.get_parameter('annotated_topic').value

        self.bridge = CvBridge()
        self.result_pub = self.create_publisher(String, result_topic, 10)
        # Full camera frame with YOLO's own boxes/labels/confidences drawn on
        # it (via Ultralytics Results.plot()) - for visual confirmation in
        # Foxglove/RViz, since /yolo_result alone is just a bare label
        # string with no way to see what the model actually saw/boxed.
        # Published every frame regardless of whether anything was detected,
        # same as any other live camera feed.
        self.annotated_pub = self.create_publisher(Image, annotated_topic, 10)
        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        if ULTRALYTICS_AVAILABLE:
            # NCNN (exported via `model.export(format='ncnn')`) rather than a
            # raw .pt checkpoint - NCNN's runtime doesn't route inference
            # through torch's BLAS/CUDA backend at all, which is what was
            # crashing (SIGILL, exit -4) on the Pi's Cortex-A72 with a .pt
            # model - see docs/pi-camera-vision.md "Known open issues" #1.
            self.model = YOLO(model_path, task='detect')
            self.get_logger().info(f"[mdp_yolo] Ultralytics YOLO loaded successfully from {model_path}!")
        else:
            self.model = None
            self.get_logger().warn("[mdp_yolo] Ultralytics library not installed. Simulation fallback mode active.")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        if self.model is not None:
            results = self.model(cv_image, verbose=False)
            for r in results:
                self.publish_annotated(r, msg.header)
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id].upper()
                    self.publish_detection(label)
                    return

    def publish_annotated(self, result, header):
        # result.plot() returns a BGR numpy array (same convention as the
        # cv_image this all started from) with boxes/labels/confidences
        # already drawn by Ultralytics - no manual cv2.rectangle/putText
        # needed. Reuses the original frame's header/timestamp so this
        # topic stays sync'able with /image_raw in Foxglove/rviz.
        annotated = result.plot()
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error (annotated): {e}")
            return
        out_msg.header = header
        self.annotated_pub.publish(out_msg)

    def publish_detection(self, detection: str):
        msg = String()
        msg.data = detection
        self.result_pub.publish(msg)
        self.get_logger().info(f"[mdp_yolo] YOLO Detected: {detection}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
