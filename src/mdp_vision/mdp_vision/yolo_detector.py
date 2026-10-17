#!/usr/bin/env python3
"""
Ultralytics YOLO ROS2 Detector Node.
Located in the mdp_vision package.
Subscribes to camera feed (/image_raw), runs YOLO inference,
and publishes detected target/arrow string to /yolo_result.
"""

import json
import os
import sys
import warnings

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

# Loading a CoreML model, Ultralytics checks numpy <= 2.3.5 (for its exporter) and
# tries to pip-install it: pixi's env has numpy 2.5 and no pip, so 4 lines of
# failed AutoUpdate every start. Inference is fine on 2.5 (task2_yolo26s.mlpackage
# = the .pt on all 27 test images, 2026-10-15). Never pip-install here anyway.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
# coremltools 9's own docstring (fake_palettize.py) warns on import: not ours.
warnings.filterwarnings('ignore', message=r'invalid escape sequence', category=SyntaxWarning)

try:
    from ultralytics.utils.plotting import colors
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

MODELS_DIR = os.path.join(get_package_share_directory('mdp_vision'), 'models')

# YOLO runs on the laptop (`pixi run laptop`), PyTorch weights from models/:
#   best_v5.pt        - YOLO26m trained at 640, task 1 (default)
#   task2_yolo26s.pt  - Arrow Left / Right + Bullseye, task 2 (task:=2)
# (best.pt / best_v4.pt, the older ones, were removed 2026-10-17 - git history has them)
# Switch with the `model_path` parameter (the `model:=` launch arg of
# mdp.launch.py / vision.launch.py): a file name under models/ or a full path.
# Where it runs (pick_device): an NVIDIA GPU (CUDA) when there is one; on a Mac
# the CoreML export of the same model (<name>.mlpackage next to <name>.pt, made
# once with `pixi run export-coreml`) on the Neural Engine, else the Apple GPU
# (MPS); else the CPU. `coreml:=false` keeps a Mac on the .pt (MPS).
DEFAULT_MODEL = 'best_v5.pt'


def coreml_twin(model_path: str) -> str:
    """On a Mac: <name>.mlpackage next to <name>.pt (`pixi run export-coreml`)
    when there is one - CoreML runs it on the Neural Engine. Else model_path."""
    if sys.platform != 'darwin' or not model_path.endswith('.pt'):
        return model_path
    twin = model_path[:-len('.pt')] + '.mlpackage'
    return twin if os.path.isdir(twin) else model_path


def pick_device(model_path: str):
    """(device for Ultralytics, what to log): CoreML picks its own hardware
    (None); else the NVIDIA GPU when PyTorch sees one (the laptop's CUDA build,
    pixi.toml), the Apple GPU (MPS) on a Mac, else the CPU."""
    if model_path.endswith('.mlpackage'):
        return None, 'CoreML (Apple Neural Engine / GPU)'
    try:
        import torch
        if torch.cuda.is_available():
            return 0, f'GPU (CUDA: {torch.cuda.get_device_name(0)})'
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return 'mps', 'Apple GPU (MPS)'
        return 'cpu', f'CPU (PyTorch {torch.__version__}, no CUDA / MPS)'
    except Exception:
        return 'cpu', 'CPU'


def resolve_model_path(value: str) -> str:
    """Accept either a bare model file name (e.g. 'best_v5.pt', resolved under
    the package models/ dir) or an absolute/relative path to one."""
    if os.path.isabs(value) or os.path.sep in value:
        return value
    return os.path.join(MODELS_DIR, value)


# Official MDP Target IDs (the number the tablet/professor scores by, sent as
# <Target_ID> in `TARGET,<Obstacle_ID>,<Target_ID>` - see
# docs/assessment_checklist.md C.9 / A.2). These are the same numbers that
# prefix the asset PNG filenames (e.g. 20_AlphabetA.png -> 20), so the ID is
# authoritative regardless of what a given model happens to name its classes.
#
# The model's class NAMES vary (best_v5.pt uses "Letter A"/"Number 1"/
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
    # Stop = the filled black circle (MDP image list) - best_v5.pt calls it "Circle"
    'stop': 40, 'circle': 40,
    # bullseye -> 99 (not on the MDP list: a model with a Bullseye class only)
    'bullseye': 99,
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
        self.declare_parameter('coreml', True)   # Mac: use <model>.mlpackage when it exists

        camera_topic = self.get_parameter('camera_topic').value
        model_path = resolve_model_path(self.get_parameter('model_path').value)
        if self.get_parameter('coreml').value:
            model_path = coreml_twin(model_path)
        result_topic = self.get_parameter('result_topic').value
        annotated_topic = self.get_parameter('annotated_topic').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.min_conf = float(self.get_parameter('min_confidence').value)
        self.edge_margin = int(self.get_parameter('edge_margin_px').value)

        self.bridge = CvBridge()
        self.result_pub = self.create_publisher(String, result_topic, 10)
        # Every box of every frame with the frame's own time, for analysing runs from a bag:
        # {"stamp", "w", "h", "boxes": [[id, conf, x1, y1, x2, y2]]}.
        self.boxes_pub = self.create_publisher(String, '/yolo_detections', 10)
        # Full camera frame with YOLO's own boxes/labels/confidences drawn on
        # it (via Ultralytics Results.plot()) - for visual confirmation in
        # Foxglove/RViz, since /yolo_result alone is just a bare label
        # string with no way to see what the model actually saw/boxed.
        # Published every frame regardless of whether anything was detected,
        # same as any other live camera feed. JPEG-compressed (not raw
        # Image): a raw frame is ~0.9 MB.
        self.annotated_pub = self.create_publisher(CompressedImage, annotated_topic, 10)
        # BEST_EFFORT + depth 1 (KEEP_LAST): if inference falls even slightly
        # behind the camera's frame rate, a reliable depth-10 subscription
        # queues up a backlog and the callback works through stale frames
        # forever, so what's on screen keeps drifting further behind live.
        # Depth 1 makes the middleware always hand the callback the newest
        # frame and silently drop anything older, so the feed stays live
        # instead of catching up. camera_ros's default publisher QoS is
        # RELIABLE, which is compatible with a BEST_EFFORT subscriber.
        # A topic ending in /compressed is the JPEG copy (rpi_cam_publisher
        # sends it while something subscribes) - what YOLO on the laptop reads
        # over WiFi (`pixi run laptop`): ~20 KB a frame instead of ~0.9 MB raw.
        self.compressed = camera_topic.endswith('/compressed')
        # Raw frames (same machine: Gazebo, the Pi's own camera) RELIABLE, still
        # depth 1: best effort lost nearly every ~0.9 MB sim frame since Cyclone /
        # Fast DDS replaced Zenoh - 2 of ~260 in 8 s, YOLO saw nothing (2026-10-15).
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if self.compressed else ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(CompressedImage if self.compressed else Image, camera_topic,
                                 self.image_callback, image_qos)

        if ULTRALYTICS_AVAILABLE:
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
            kwargs = {} if self.device is None else {'device': self.device}   # CoreML: its own choice
            results = self.model(cv_image, verbose=False, conf=self.min_conf, **kwargs)
            for r in results:
                self.publish_boxes(r, msg.header, cv_image.shape)
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

    def publish_boxes(self, result, header, shape):
        if not len(result.boxes):
            return
        boxes = []
        for box in result.boxes:
            name = self.model.names[int(box.cls[0])]
            tid = label_to_target_id(name)
            boxes.append([str(tid if tid is not None else name.upper()), round(float(box.conf[0]), 3)]
                         + [round(float(v), 1) for v in box.xyxy[0]])
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self.boxes_pub.publish(String(data=json.dumps(
            {'stamp': stamp, 'w': shape[1], 'h': shape[0], 'boxes': boxes})))

    def pick_box(self, result, shape):
        """The box to report (see min_confidence / edge_margin_px): the
        biggest whole one; a cut-off one only when nothing whole is seen; within 10 %
        in size, the more confident. None when nothing is seen."""
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
