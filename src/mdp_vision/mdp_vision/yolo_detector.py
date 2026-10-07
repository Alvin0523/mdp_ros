#!/usr/bin/env python3
"""
Ultralytics YOLO ROS2 Detector Node.
Located in the mdp_vision package.
Subscribes to camera feed (/image_raw), runs YOLO inference,
and publishes detected target/arrow string to /yolo_result.
"""

import os

import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge

try:
    from ultralytics.utils.plotting import colors
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# The `_ncnn_model` suffix is required, not a naming choice - ultralytics'
# AutoBackend detects the model format from the directory name itself
# (every export format has its own required suffix: *_ncnn_model/,
# *_saved_model/, *_openvino_model/, ...), not from the files inside it.
MODELS_DIR = os.path.join(get_package_share_directory('mdp_vision'), 'models')

# Default to the latest MDP-trained model (classes = Arrow/Letter/Number/
# Circle - the actual task symbols). Available models:
#   mdp_v2_ncnn_model  - latest MDP model (default)
#   mdp_v1_ncnn_model  - older MDP model (kept for comparison)
# Switch with the `model_path` parameter (see vision.launch.py /
# mdp.launch.py `model:=` launch arg), which accepts either a bare
# model-dir name under models/ or an absolute path.
DEFAULT_MODEL = 'mdp_v2_ncnn_model'


def pick_device(model_path: str):
    """(device for Ultralytics, what to log). A .pt model runs on the NVIDIA GPU
    when PyTorch sees one (the laptop's CUDA build, pixi.toml), else the CPU; an
    NCNN model always runs on the CPU (NCNN's own runtime)."""
    if not model_path.endswith('.pt'):
        return 'cpu', 'CPU (NCNN)'
    try:
        import torch
        if torch.cuda.is_available():
            return 0, f'GPU (CUDA: {torch.cuda.get_device_name(0)})'
        return 'cpu', f'CPU (PyTorch {torch.__version__}, no CUDA)'
    except Exception:
        return 'cpu', 'CPU'


def resolve_model_path(value: str) -> str:
    """Accept either a bare model name (e.g. 'mdp_v2_ncnn_model', resolved under
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
# The model's class NAMES vary (mdp_v2_ncnn_model uses "Letter A"/"Number 1"/
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
        self.declare_parameter('jpeg_quality', 80)
        # Which box to report when YOLO sees several (pick_box): below
        # min_confidence is ignored (Ultralytics' own default was 0.25); boxes
        # within edge_margin_px of the frame edge are cut off, used only when
        # nothing else is seen; of the rest the BIGGEST - the symbols are all one
        # size, so the closest one looks biggest (a neighbouring block further
        # away was read instead of the target, sim 2026-09-30).
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('edge_margin_px', 3)

        camera_topic = self.get_parameter('camera_topic').value
        model_path = resolve_model_path(self.get_parameter('model_path').value)
        result_topic = self.get_parameter('result_topic').value
        annotated_topic = self.get_parameter('annotated_topic').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.min_conf = float(self.get_parameter('min_confidence').value)
        self.edge_margin = int(self.get_parameter('edge_margin_px').value)

        self.bridge = CvBridge()
        self.result_pub = self.create_publisher(String, result_topic, 10)
        # Full camera frame with YOLO's own boxes/labels/confidences drawn on
        # it (via Ultralytics Results.plot()) - for visual confirmation in
        # Foxglove/RViz, since /yolo_result alone is just a bare label
        # string with no way to see what the model actually saw/boxed.
        # Published every frame regardless of whether anything was detected,
        # same as any other live camera feed. JPEG-compressed (not raw
        # Image) - inference itself is the real bottleneck on this hardware
        # (~0.7 fps measured on a Pi 4B with the NCNN model), but a raw
        # frame every ~1.4s is still ~0.9MB each; publishing compressed
        # avoids adding unnecessary bandwidth/decode cost on top of that.
        self.annotated_pub = self.create_publisher(CompressedImage, annotated_topic, 10)
        # BEST_EFFORT + depth 1 (KEEP_LAST): if inference falls even slightly
        # behind the camera's frame rate, a reliable depth-10 subscription
        # queues up a backlog and the callback works through stale frames
        # forever, so what's on screen keeps drifting further behind live.
        # Depth 1 makes the middleware always hand the callback the newest
        # frame and silently drop anything older, so the feed stays live
        # instead of catching up. camera_ros's default publisher QoS is
        # RELIABLE, which is compatible with a BEST_EFFORT subscriber.
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        # A topic ending in /compressed is the JPEG copy (rpi_cam_publisher
        # sends it while something subscribes) - what YOLO on the laptop reads
        # over WiFi (`pixi run laptop`): ~20 KB a frame instead of ~0.9 MB raw.
        self.compressed = camera_topic.endswith('/compressed')
        self.create_subscription(CompressedImage if self.compressed else Image, camera_topic,
                                 self.image_callback, image_qos)

        if ULTRALYTICS_AVAILABLE:
            # NCNN (exported via `model.export(format='ncnn')`) rather than a
            # raw .pt checkpoint - NCNN's runtime doesn't route inference
            # through torch's BLAS/CUDA backend at all, which is what was
            # crashing (SIGILL, exit -4) on the Pi's Cortex-A72 with a .pt
            # model - see docs/pi-camera-vision.md "Known open issues" #1.
            self.model = YOLO(model_path, task='detect')
            self.device, where = pick_device(model_path)
            self.get_logger().info(f"YOLO      model loaded: {os.path.basename(model_path)} on {where}")
        else:
            self.model = None
            self.get_logger().warn("YOLO      ultralytics not installed - no detections")

    def image_callback(self, msg):
        try:
            if self.compressed:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"YOLO      bad image: {e}")
            return

        if self.model is not None:
            results = self.model(cv_image, verbose=False, device=self.device, conf=self.min_conf)
            for r in results:
                box = self.pick_box(r, cv_image.shape)
                self.publish_annotated(r, msg.header, box)
                if box is None:
                    return
                class_name = self.model.names[int(box.cls[0])]
                target_id = label_to_target_id(class_name)
                if target_id is None:
                    self.get_logger().warn(f"YOLO      no MDP Target ID for class {class_name!r}")
                    target_id = class_name.upper()
                self.publish_detection(str(target_id), class_name)
                return

    def pick_box(self, result, shape):
        """The box to report (see min_confidence / edge_margin_px): the biggest
        whole one; a cut-off one only when nothing whole is seen; within 10 % in
        size, the more confident. None when nothing is seen."""
        h, w = shape[:2]
        m = self.edge_margin
        whole, cut = [], []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            edge = x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m
            (cut if edge else whole).append(((x2 - x1) * (y2 - y1), float(box.conf[0]), box))
        boxes = whole or cut
        if not boxes:
            return None
        biggest = max(a for a, _, _ in boxes)
        close = [b for b in boxes if b[0] >= 0.9 * biggest]       # as big as the biggest, within 10 %
        chosen = max(close, key=lambda b: b[1])
        if len(whole) + len(cut) > 1:
            name = lambda b: self.model.names[int(b[2].cls[0])]
            others = ', '.join(name(b) for b in whole + cut if b is not chosen)
            self.get_logger().debug(f"YOLO      picked {name(chosen)} (biggest{'' if whole else ', cut off'}) "
                                    f"over {others}")
        return chosen[2]

    def publish_annotated(self, result, header, chosen=None):
        # Drawing boxes/labels and re-encoding a full frame costs real CPU on
        # the same core doing inference - skip it entirely when nobody's
        # actually subscribed (e.g. no Foxglove/rviz open during a real run).
        if self.annotated_pub.get_subscription_count() == 0:
            return

        # Every box labelled 'Number 1 (11) 0.90' - class, MDP target id,
        # confidence - in YOLO's own class colour; the one sent on /yolo_result
        # thick and green. Same frame and header as the image YOLO read.
        annotated = result.orig_img.copy()
        h = annotated.shape[0]
        # (A new object per box on every pass over result.boxes: match by coordinates.)
        is_sent = lambda b: chosen is not None and bool((b.xyxy == chosen.xyxy).all())
        boxes = sorted(result.boxes, key=is_sent)   # the sent box drawn last, on top
        for box in boxes:
            sent = is_sent(box)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            colour = (0, 200, 0) if sent else colors(int(box.cls[0]), True)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 4 if sent else 2)
            label = self.box_label(box)
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 6 if y1 - th - 10 >= 0 else min(h - 4, y2 + th + 8)   # above the box, else below
            lx = max(0, min(x1, annotated.shape[1] - tw - 8))                 # kept inside the frame
            cv2.rectangle(annotated, (lx, ty - th - 5), (lx + tw + 8, ty + base), colour, -1)
            cv2.putText(annotated, label, (lx + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ok, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        out_msg = CompressedImage()
        out_msg.header = header
        out_msg.format = 'jpeg'
        out_msg.data = jpeg.tobytes()
        self.annotated_pub.publish(out_msg)

    def box_label(self, box) -> str:
        """'Number 1 (11) 0.90': class name, MDP target id, confidence."""
        name = self.model.names[int(box.cls[0])]
        tid = label_to_target_id(name)
        return f"{name}{f' ({tid})' if tid is not None else ''} {float(box.conf[0]):.2f}"

    def publish_detection(self, target_id: str, class_name: str = None):
        # /yolo_result carries the official MDP Target ID string (e.g. "20"),
        # which task1_runner forwards verbatim as TARGET,<obs>,<target_id>.
        msg = String()
        msg.data = target_id
        self.result_pub.publish(msg)
        if class_name is not None:
            self.get_logger().debug(f"YOLO      {class_name} -> {target_id}")
        else:
            self.get_logger().debug(f"YOLO      {target_id}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise        # a real error; after Ctrl+C it is only the shutdown racing a callback
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # already stopping: ignore a second Ctrl+C
        try:
            node.destroy_node()
        except (Exception, KeyboardInterrupt):
            pass         # ROS already shut down / a second Ctrl+C
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
