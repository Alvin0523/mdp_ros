"""Finding task 2's bullseyes by template matching (OpenCV, no model): a stand-in
until YOLO is trained on them (class 99). bullseye_matcher.py runs it on the
camera and publishes the boxes like YOLO's /yolo_detections.

The template is the printed bullseye itself (mdp_description
models/symbols/99_Bullseye.png - concentric squares). It is matched at every
size from min_px to max_px: small sizes with the template shrunk on the full
frame, big ones with the FRAME shrunk instead (the template kept at BASE_PX),
so each size costs about the same.
"""

from typing import List, Tuple

import cv2
import numpy as np

BASE_PX = 24          # template side for sizes above it (the frame is shrunk instead)
SCALE_STEP = 1.12     # one size to the next


def load_template(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def sizes(min_px: float, max_px: float) -> List[float]:
    out, s = [], float(min_px)
    while s <= max_px:
        out.append(s)
        s *= SCALE_STEP
    return out


def find_bullseyes(gray: np.ndarray, template: np.ndarray, min_px: float = 10.0, max_px: float = 140.0,
                   threshold: float = 0.6, max_boxes: int = 6) -> List[Tuple[float, float, float, float, float]]:
    """(score, x1, y1, x2, y2) of the bullseyes in a grayscale frame, best first,
    overlapping ones merged (the best kept)."""
    h, w = gray.shape[:2]
    found = []
    for s in sizes(min_px, max_px):
        if s <= BASE_PX:
            frame, k, t = gray, 1.0, max(4, int(round(s)))
        else:
            k = BASE_PX / s
            frame = cv2.resize(gray, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
            t = BASE_PX
        if frame.shape[0] < t or frame.shape[1] < t:
            break
        tmpl = cv2.resize(template, (t, t), interpolation=cv2.INTER_AREA)
        score = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(score >= threshold)
        for y, x in zip(ys, xs):
            found.append((float(score[y, x]), x / k, y / k, (x + t) / k, (y + t) / k))
    found.sort(key=lambda b: -b[0])
    kept = []
    for b in found:
        if all(_iou(b, k) < 0.2 and not _inside(b, k) for k in kept):
            kept.append(b)
            if len(kept) >= max_boxes:
                break
    return kept


def _iou(a, b) -> float:
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def _inside(a, b) -> bool:
    """a's centre inside b, or b's inside a (the inner squares match a smaller size)."""
    def centre_in(p, q):
        cx, cy = (p[1] + p[3]) / 2.0, (p[2] + p[4]) / 2.0
        return q[1] <= cx <= q[3] and q[2] <= cy <= q[4]
    return centre_in(a, b) or centre_in(b, a)
