"""Task 2: what the camera's boxes say (ROS-free) - which arrow is which, and the
bar's width from a bullseye.

Boxes come as YOLO publishes them on /yolo_detections (and the laptop's
bullseye_matcher on /bullseye_detections, in the same format):
    {"stamp": s, "w": 640, "h": 480, "boxes": [[id, conf, x1, y1, x2, y2], ...]}
ids: '39' arrow left, '38' arrow right, '99' bullseye (MDP target ids).

Every symbol is printed the same size (SYMBOL_M square), so a box's height says
how far away it is, and its centre which way: a pinhole camera with the
horizontal view hfov, looking along the car's heading (task 2's camera).
"""

import math
from typing import List, Optional, Tuple

from mdp_bringup.utils.obstacle_layout import SYMBOL_M

ARROW_IDS = {'39': 'LEFT', '38': 'RIGHT'}
BULLSEYE_ID = '99'
# A box's range from its size must agree with the geometry's within this ratio: a
# bullseye elsewhere (obstacle 1's sides) or a false match is a different size.
RANGE_RATIO = (0.75, 1.33)
MIN_ASPECT = 0.6              # w / h: a face seen nearly square-on (not the bar's end faces, edge-on)
ARROW_2_MARGIN_M = 0.35       # an arrow this much further than obstacle 1's face is arrow 2 (d2 >= 0.6 m)
# YOLO's arrow box is the arrow, not the whole printed card: 33 px where the card is
# 44 (best_v5.pt, sim 2026-10-14). The runner measures it at home (obstacle 1's
# distance known) - this until then. Read as the full card it put arrow 1 a third
# further away: at d1 = 1.0 m arrow 1 was taken for arrow 2.
ARROW_BOX_SCALE = 0.75
CONFIDENT = 0.70              # a bullseye box this sure is used for the bar's distance
MIN_BULLSEYE_PX = 16          # the bar's face is at most ~2.1 m away while measuring: >= 16 px


def focal_px(image_w: float, hfov: float) -> float:
    return (image_w / 2.0) / math.tan(hfov / 2.0)


def box_range(box, image_w: float, hfov: float, scale: float = 1.0) -> float:
    """Distance along the camera's axis to a symbol from its box height (m). scale:
    the box's size as a share of the printed card (ARROW_BOX_SCALE for an arrow)."""
    h = max(1.0, box[5] - box[3])
    return SYMBOL_M * scale * focal_px(image_w, hfov) / h


def box_bearing(box, image_w: float, hfov: float) -> float:
    """Angle of the box's centre off the camera's axis, rad, + left."""
    u = (box[2] + box[4]) / 2.0
    return math.atan2(image_w / 2.0 - u, focal_px(image_w, hfov))


def parse(data: str):
    """(stamp, w, h, boxes) from a /yolo_detections message; boxes [id, conf, x1, y1, x2, y2]."""
    import json
    d = json.loads(data)
    boxes = [[str(b[0])] + [float(v) for v in b[1:6]] for b in d.get('boxes', [])]
    return float(d['stamp']), float(d['w']), float(d['h']), boxes


def classify_arrow(box, image_w: float, hfov: float, cam_x: float, obstacle_1_face_x: Optional[float],
                   scale: float = ARROW_BOX_SCALE) -> int:
    """0 = arrow 1 (obstacle 1's), 1 = arrow 2 (the bar's): arrow 2 is the one well
    beyond obstacle 1's face. Obstacle 1 unknown (or passed): arrow 1 if the camera
    is short of it."""
    if obstacle_1_face_x is None:
        return 0
    ahead = obstacle_1_face_x - cam_x
    if ahead <= 0.0:
        return 1                 # past obstacle 1's face: its arrow is out of view
    return 1 if box_range(box, image_w, hfov, scale) > ahead + ARROW_2_MARGIN_M else 0


def arrow_scale(box, image_w: float, hfov: float, distance: float) -> float:
    """The arrow box's size as a share of the card, from one seen `distance` away."""
    return (box[5] - box[3]) * distance / (SYMBOL_M * focal_px(image_w, hfov))


def bullseye_position(box, image_w: float, hfov: float, cam: Tuple[float, float, float]):
    """Where a symbol is (map x, y) from its box alone: the distance from its size,
    the direction from its centre. Coarser than bar_half_width()'s (a pixel of size
    is 2-4 % of the distance at 0.5-1.5 m) - the check on the ultrasonic, which
    looks straight past a bar narrower than the lane is wide (sim 2026-10-14: a
    40 cm bar read 25 cm too far)."""
    theta = box_bearing(box, image_w, hfov)
    t = box_range(box, image_w, hfov) / max(0.2, math.cos(theta))      # along the ray
    phi = cam[2] + theta
    return cam[0] + t * math.cos(phi), cam[1] + t * math.sin(phi)


def bar_half_width(box, image_w: float, hfov: float, cam: Tuple[float, float, float], face_x: float,
                   centre_y: float, block_m: float = 0.10) -> Optional[float]:
    """Half the bar's width from one bullseye on its near face, or None if the box
    does not fit that (wrong size for the distance, seen edge-on, behind the camera).

    The bullseyes on the bar's face sit on its edge-most block, centred, so the
    bar's end is block_m / 2 further out than the bullseye. The bar's face is at
    face_x (the ultrasonic), so only the bullseye's direction is needed - not its
    size, which at 2-3 m away is a few px and +-1 px off (sim 2026-10-14:
    carpark estimates up to 7 cm off; at checkpoint 1 0.2-0.3 cm)."""
    w, h = box[4] - box[2], box[5] - box[3]
    if w <= 0 or h <= 0 or w / h < MIN_ASPECT:
        return None
    cx, cy, yaw = cam
    phi = yaw + box_bearing(box, image_w, hfov)
    if math.cos(phi) < 0.2:
        return None
    t = (face_x - cx) / math.cos(phi)               # along the ray to the face
    if t <= 0.05:
        return None
    axial = t * math.cos(phi - yaw)                 # its distance along the camera's axis
    ratio = box_range(box, image_w, hfov) / axial
    if not RANGE_RATIO[0] <= ratio <= RANGE_RATIO[1]:
        return None
    y = cy + t * math.sin(phi)
    return abs(y - centre_y) + block_m / 2.0


def confident(box) -> bool:
    """A bullseye box sure and big enough to place the bar by (the matcher's
    smallest-size matches on textures were all false, sim 2026-10-14)."""
    return box[1] >= CONFIDENT and box[5] - box[3] >= MIN_BULLSEYE_PX


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
