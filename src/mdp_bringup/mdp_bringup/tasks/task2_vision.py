"""Task 2: what the camera's boxes say (ROS-free) - which arrow is which, and the
bar's width from its bullseyes (and arrow 2) in the picture.

Boxes come as YOLO publishes them on /yolo_detections (and the laptop's
bullseye_matcher on /bullseye_detections, in the same format):
    {"stamp": s, "w": 640, "h": 480, "boxes": [[id, conf, x1, y1, x2, y2], ...]}
ids: '39' arrow left, '38' arrow right, '99' bullseye (MDP target ids).

Every symbol is printed the same size (SYMBOL_M square; a bullseye is boxed by its
outer white square, BULLSEYE_RIM_M - its black edge is lost against the black
obstacle), so a box's height says
how far away it is, and its centre which way: a pinhole camera with the
horizontal view hfov, looking along the car's heading (task 2's camera).
"""

import math
from typing import List, Optional, Tuple

from mdp_bringup.utils.obstacle_layout import BULLSEYE_FROM_END_M, BULLSEYE_RIM_M, SYMBOL_M

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
MIN_BULLSEYE_PX = 12          # the bar's face is at most ~2.1 m away while measuring: the rim >= 12 px


def focal_px(image_w: float, hfov: float) -> float:
    return (image_w / 2.0) / math.tan(hfov / 2.0)


def box_range(box, image_w: float, hfov: float, scale: float = 1.0, size: float = SYMBOL_M) -> float:
    """Distance along the camera's axis to a symbol from its box height (m). size: what
    is boxed (the card, BULLSEYE_RIM_M for a bullseye); scale: the box's share of it
    (ARROW_BOX_SCALE for an arrow)."""
    h = max(1.0, box[5] - box[3])
    return size * scale * focal_px(image_w, hfov) / h


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
    t = box_range(box, image_w, hfov, size=BULLSEYE_RIM_M) / max(0.2, math.cos(theta))   # along the ray
    phi = cam[2] + theta
    return cam[0] + t * math.cos(phi), cam[1] + t * math.sin(phi)


def bar_half_width(box, image_w: float, hfov: float, cam: Tuple[float, float, float], face_x: float,
                   centre_y: float, end_m: float = BULLSEYE_FROM_END_M) -> Optional[float]:
    """Half the bar's width from one bullseye on its near face, or None if the box
    does not fit that (wrong size for the distance, seen edge-on, behind the camera).

    The bullseyes on the bar's face sit near its ends: the bar's end is end_m
    further out than the bullseye. The bar's face is at
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
    ratio = box_range(box, image_w, hfov, size=BULLSEYE_RIM_M) / axial
    if not RANGE_RATIO[0] <= ratio <= RANGE_RATIO[1]:
        return None
    y = cy + t * math.sin(phi)
    return abs(y - centre_y) + end_m


def plausible_bullseye(box, image_w: float, hfov: float, z: float, yaw: float = 0.0) -> bool:
    """A bullseye box seen square-on and the right size for a face z ahead (the
    camera turned yaw off square to it)."""
    w, h = box[4] - box[2], box[5] - box[3]
    if w <= 0 or h <= 0 or w / h < MIN_ASPECT or z <= 0.05:
        return False
    theta = box_bearing(box, image_w, hfov)
    if math.cos(yaw + theta) < 0.2:
        return False
    axial = z * math.cos(theta) / math.cos(yaw + theta)
    return RANGE_RATIO[0] <= box_range(box, image_w, hfov, size=BULLSEYE_RIM_M) / axial <= RANGE_RATIO[1]


def lateral(box, image_w: float, hfov: float, z: float, yaw: float = 0.0) -> float:
    """Where a box's centre is across a face z ahead (m, + left of the camera), the
    camera turned yaw (+ left) off square to the face."""
    return z * math.tan(yaw + box_bearing(box, image_w, hfov))


def bar_half_width_gap(arrow_box, bullseye_box, image_w: float, hfov: float, z: float, yaw: float = 0.0,
                       end_m: float = BULLSEYE_FROM_END_M, gap_scale: float = 1.0) -> float:
    """Half the bar's width from the gap between arrow 2 (the face's centre) and a
    bullseye (mid its edge-most block) in the SAME frame, z = the face's distance
    (ultrasonic). A heading error turns both the same, so the gap - and the width -
    barely feels it, and the position across the course not at all. The heading
    (yaw) only corrects perspective: 3 deg wrong is ~ z * 3 deg * (tan^2 of one's
    angle off square to the face - the other's): ~2 cm of width at checkpoint 1,
    d2 = 0.6 m (arrow 2 32 deg off), less further away; the ray from the pose
    (bar_half_width()) 5 cm, and 2 cm per cm the position is off across. gap_scale:
    the lens' correction (navigation.yaml bar_gap_scale)."""
    gap = abs(lateral(bullseye_box, image_w, hfov, z, yaw) - lateral(arrow_box, image_w, hfov, z, yaw))
    return gap * gap_scale + end_m


def gap_widths(frames, hfov: float, side: float, gap_scale: float = 1.0):
    """The bar's widths from frames [(arrow 2 box or None, [bullseye boxes], image w, z, yaw)]:
    [(bullseye box height, width)]: one end and arrow 2 (mid-bar) - twice their gap.
    Bullseyes on the car's side (`side` +1 left / -1 right) first - the far end's only
    if none. (Both ends' gap without the arrow: dropped 2026-10-15, arrow + bullseye only.)"""
    near, far = [], []
    for arrow, boxes, w, z, yaw in frames:
        boxes = [b for b in boxes if confident(b) and plausible_bullseye(b, w, hfov, z, yaw)]
        if not boxes:
            continue
        xs = sorted((lateral(b, w, hfov, z, yaw), b) for b in boxes)
        if arrow is None:
            continue
        ya = lateral(arrow, w, hfov, z, yaw)
        for y, b in xs[:1] if len(xs) == 1 else (xs[0], xs[-1]):
            est = (b[5] - b[3], 2.0 * bar_half_width_gap(arrow, b, w, hfov, z, yaw, gap_scale=gap_scale))
            (near if (y - ya) * side > 0 else far).append(est)
    return near or far


def pick_width(sized, width_range):
    """(width or None, widths used) from [(bullseye box height, width)]: plausible ones,
    the nearest half (biggest boxes - a far frame's pixel is more of the width; sim
    2026-10-14: up to 9 cm off at d2 = 1.5 with every frame), median, clamped."""
    lo, hi = width_range
    sized = sorted((s for s in sized if lo - 0.05 <= s[1] <= hi + 0.05), reverse=True)
    widths = [wd for _, wd in sized[:max(1, (len(sized) + 1) // 2)]]
    width = median(widths)
    return (None if width is None else min(hi, max(lo, width))), widths


def merge_duplicates(boxes):
    """One box per bullseye: YOLO's (task2_yolo26s.pt boxes them too) and the
    matcher's of the same frame overlap - YOLO's kept ('yolo' as the box's 7th item),
    then the more confident."""
    kept = []
    for b in sorted(boxes, key=lambda b: (len(b) > 6 and b[6] == 'yolo', b[1]), reverse=True):
        u, v = (b[2] + b[4]) / 2.0, (b[3] + b[5]) / 2.0
        if not any(k[2] <= u <= k[4] and k[3] <= v <= k[5] for k in kept):
            kept.append(b)
    return kept


def bar_width(bullseyes, arrows_2, hfov: float, face_x: float, centre_y: float, side: float, width_range,
              pair_s: float = 0.02, gap_scale: float = 1.0):
    """The bar's width from the frames seen near checkpoint 1: (width or None, widths,
    'gap' / 'pose'). bullseyes [(stamp, box, image w, camera pose)], arrows_2 [(stamp,
    box)] - YOLO's and the matcher's boxes of one frame carry the frame's stamp.
    The gap method (gap_widths(): no position across, the heading only for
    perspective) first; else
    bar_half_width() from the camera's pose."""
    frames = {}
    for stamp, box, w, cam in bullseyes:
        frames.setdefault(stamp, (w, cam, []))[2].append(box)
    frames = {t: (w, cam, merge_duplicates(boxes)) for t, (w, cam, boxes) in frames.items()}
    gap_frames = []
    for stamp, (w, cam, boxes) in frames.items():
        near = [a for t, a in arrows_2 if abs(t - stamp) <= pair_s]
        arrow = max(near, key=lambda a: a[1]) if near else None
        gap_frames.append((arrow, boxes, w, face_x - cam[0], cam[2]))
    width, widths = pick_width(gap_widths(gap_frames, hfov, side, gap_scale), width_range)
    if width is not None:
        return width, widths, 'gap'
    sized = []
    for w, cam, boxes in frames.values():
        for box in boxes:
            if not confident(box):
                continue
            half = bar_half_width(box, w, hfov, cam, face_x, centre_y)
            if half is not None:
                sized.append((box[5] - box[3], 2.0 * half))
    width, widths = pick_width(sized, width_range)
    return width, widths, 'pose'


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
