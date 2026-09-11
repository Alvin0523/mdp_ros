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
MODELS_DIR = os.path.join(get_package_share_directory('mdp_yolo'), 'models')

# Default to the latest MDP-trained model (classes = Arrow/Letter/Number/
# Circle - the actual task symbols), NOT the stock yolo26n COCO model
# (person/car/...) which cannot detect any MDP symbol. Available models:
#   best_ncnn_model_v2  - latest MDP model (default)
#   best_ncnn_model_v1  - older MDP model (kept for comparison)
#   yolo26n_ncnn_model  - stock YOLO26n COCO net (debug only)
# Switch with the `model_path` parameter (see vision.launch.py /
# task2_sim.launch.py `model:=` launch arg), which accepts either a bare
# model-dir name under models/ or an absolute path.
DEFAULT_MODEL = 'best_ncnn_model_v2'


def resolve_model_path(value: str) -> str:
    """Accept either a bare model name (e.g. 'best_ncnn_model', resolved under
    the package models/ dir) or an absolute/relative path to a model dir."""
    if os.path.isabs(value) or os.path.sep in value:
        return value
    return os.path.join(MODELS_DIR, value)


# Official MDP Target IDs (the number the tablet/professor scores by, sent as
# <Target_ID> in `TARGET,<Obstacle_ID>,<Target_ID>` - see
# docs/assessment_checklist.md C.9 / A.2). These are the same numbers that
# prefix the asset PNG filenames (e.g. 20_AlphabetA.png -> 20), so the ID is
# authoritative regardless of what a given model happens to name its classes.
#
# The model's class NAMES vary (best_ncnn_model uses "Letter A"/"Number 1"/
# "Arrow Up"/"Circle"; a raw dataset export might use "AlphabetA"/"One"/...),
# so we map by a normalised key (lowercased, non-alphanumerics stripped)
# rather than the exact string. Publishing the ID - not the name - is what
# lets task1_runner forward a meaningful <Target_ID> to the tablet.
MDP_TARGET_IDS = {
    # digits 1-9 -> 11-19
    'one': 11, 'number1': 11, 'two': 12, 'number2': 12, 'three': 13, 'number3': 13,
    'four': 14, 'number4': 14, 'five': 15, 'number5': 15, 'six': 16, 'number6': 16,
    'seven': 17, 'number7': 17, 'eight': 18, 'number8': 18, 'nine': 19, 'number9': 19,
    # letters A-H -> 20-27
    'alphabeta': 20, 'lettera': 20, 'alphabetb': 21, 'letterb': 21,
    'alphabetc': 22, 'letterc': 22, 'alphabetd': 23, 'letterd': 23,
    'alphabete': 24, 'lettere': 24, 'alphabetf': 25, 'letterf': 25,
    'alphabetg': 26, 'letterg': 26, 'alphabeth': 27, 'letterh': 27,
    # letters S,T,U,V,W,X,Y,Z -> 28-35
    'alphabets': 28, 'letters': 28, 'alphabett': 29, 'lettert': 29,
    'alphabetu': 30, 'letteru': 30, 'alphabetv': 31, 'letterv': 31,
    'alphabetw': 32, 'letterw': 32, 'alphabetx': 33, 'letterx': 33,
    'alphabety': 34, 'lettery': 34, 'alphabetz': 35, 'letterz': 35,
    # arrows + stop -> 36-40
    'arrowup': 36, 'up': 36, 'uparrow': 36,
    'arrowdown': 37, 'down': 37, 'downarrow': 37,
    'arrowright': 38, 'right': 38, 'rightarrow': 38,
    'arrowleft': 39, 'left': 39, 'leftarrow': 39,
    'stop': 40,
    # bullseye -> 99
    'bullseye': 99, 'circle': 99,
}


def _normalise_label(name: str) -> str:
    """Lowercase and strip everything but a-z0-9, so 'Letter A', 'letter_a',
    'AlphabetA' all collapse to 'lettera'/'alphabeta' consistently."""
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def label_to_target_id(name: str):
    """Model class name -> official MDP Target ID (int), or None if unknown."""
    return MDP_TARGET_IDS.get(_normalise_label(name))

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('result_topic', '/yolo_result')
        self.declare_parameter('annotated_topic', '/yolo_result/image_annotated')

        camera_topic = self.get_parameter('camera_topic').value
        model_path = resolve_model_path(self.get_parameter('model_path').value)
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

    def publish_detection(self, target_id: str, class_name: str = None):
        # /yolo_result carries the official MDP Target ID string (e.g. "20"),
        # which task1_runner forwards verbatim as TARGET,<obs>,<target_id>.
        msg = String()
        msg.data = target_id
        self.result_pub.publish(msg)
        if class_name is not None:
            self.get_logger().info(
                f"[mdp_yolo] YOLO Detected: {class_name} -> Target ID {target_id}")
        else:
            self.get_logger().info(f"[mdp_yolo] YOLO Detected Target ID {target_id}")

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
