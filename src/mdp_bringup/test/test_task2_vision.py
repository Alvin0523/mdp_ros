"""Task 2's camera geometry (tasks/task2_vision.py): a bullseye projected through
the pinhole model from a known bar comes back as the bar's width; arrows are
told apart by distance; wrong-sized and edge-on boxes are refused."""
import math

import pytest

from mdp_bringup.tasks import task2_vision as V
from mdp_bringup.utils.obstacle_layout import BULLSEYE_RIM_M, SYMBOL_M

W, H, HFOV = 640.0, 480.0, 1.089
CY = 1.20


def project(cam, point, size=BULLSEYE_RIM_M):
    """The box a symbol centred at `point` (on a face across the course) makes."""
    cx, cy, yaw = cam
    dx, dy = point[0] - cx, point[1] - cy
    z = dx * math.cos(yaw) + dy * math.sin(yaw)          # along the camera's axis
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)        # + left
    f = V.focal_px(W, HFOV)
    u = W / 2.0 - f * lat / z
    half = f * size / z / 2.0
    return ['99', 0.9, u - half, H / 2 - half, u + half, H / 2 + half]


@pytest.mark.parametrize('d2, width, side', [(0.60, 0.40, 1), (0.60, 0.80, -1), (1.00, 0.60, 1), (1.50, 0.80, 1),
                                             (1.50, 1.20, -1)])
def test_width_from_the_near_bullseye(d2, width, side):
    x1 = 1.25
    face_x = x1 + 0.05 + d2
    cam = (x1 + 0.171 - 0.10, CY + side * 0.25, 0.0)     # just before checkpoint 1, straight
    bullseye = (face_x, CY + side * (width / 2.0 - 0.057))
    box = project(cam, bullseye)
    half = V.bar_half_width(box, W, HFOV, cam, face_x, CY)
    assert 2 * half == pytest.approx(width, abs=0.005)


def test_heading_error_moves_the_estimate_little():
    cam = (1.30, CY + 0.25, math.radians(3.0))
    face_x = 1.90
    box = project(cam, (face_x, CY + 0.25))
    # The same box read with the car thought straight: off by ~ distance x 3 deg.
    half = V.bar_half_width(box, W, HFOV, (cam[0], cam[1], 0.0), face_x, CY)
    assert abs(2 * half - 0.60) < 0.07


def test_wrong_size_and_edge_on_boxes_are_refused():
    cam, face_x = (1.30, CY + 0.25, 0.0), 1.90
    box = project(cam, (face_x, CY + 0.25), size=BULLSEYE_RIM_M * 3)   # far too big for that distance
    assert V.bar_half_width(box, W, HFOV, cam, face_x, CY) is None
    box = project(cam, (face_x, CY + 0.25))
    box[2] = box[2] + 0.6 * (box[4] - box[2])                        # narrow: a face seen edge-on
    assert V.bar_half_width(box, W, HFOV, cam, face_x, CY) is None


def test_arrow_1_and_2_by_distance():
    cam_x, face1 = 0.90, 1.20                                       # obstacle 1's face 0.3 m ahead
    near = project((cam_x, CY, 0.0), (face1, CY), size=SYMBOL_M)
    far = project((cam_x, CY, 0.0), (face1 + 0.1 + 0.6, CY + 0.1), size=SYMBOL_M)
    assert V.classify_arrow(near, W, HFOV, cam_x, face1) == 0
    assert V.classify_arrow(far, W, HFOV, cam_x, face1) == 1
    assert V.classify_arrow(near, W, HFOV, 1.25, face1) == 1        # past obstacle 1: only the bar's
    assert V.classify_arrow(near, W, HFOV, cam_x, None) == 0


def test_parse():
    stamp, w, h, boxes = V.parse('{"stamp": 1.5, "w": 640, "h": 480, "boxes": [["39", 0.9, 1, 2, 3, 4]]}')
    assert (stamp, w, h) == (1.5, 640.0, 480.0) and boxes == [['39', 0.9, 1.0, 2.0, 3.0, 4.0]]


@pytest.mark.parametrize('ahead, across', [(0.6, 0.0), (1.0, 0.25), (1.6, -0.45)])
def test_bullseye_position_from_its_box(ahead, across):
    # The check on the ultrasonic (which looks past a narrow bar): within a few % of the distance.
    cam = (1.30, CY + 0.25, math.radians(4.0))
    point = (cam[0] + ahead, cam[1] + across)
    x, y = V.bullseye_position(project(cam, point), W, HFOV, cam)
    assert x == pytest.approx(point[0], abs=0.03 * ahead + 0.005)
    assert y == pytest.approx(point[1], abs=0.02)


def test_arrow_1_at_1_m_is_not_arrow_2():
    # YOLO's arrow box is ~75 % of the card: read as the full card, arrow 1 at
    # d1 = 1.0 m looked a third further and was taken for arrow 2 (sim 2026-10-14).
    cam_x, face1 = 0.50, 1.60
    near = project((cam_x, CY, 0.0), (face1, CY), size=SYMBOL_M * V.ARROW_BOX_SCALE)
    far = project((cam_x, CY, 0.0), (face1 + 0.1 + 0.6, CY), size=SYMBOL_M * V.ARROW_BOX_SCALE)
    assert V.classify_arrow(near, W, HFOV, cam_x, face1, scale=1.0) == 1      # the old reading: wrong
    assert V.classify_arrow(near, W, HFOV, cam_x, face1) == 0
    assert V.classify_arrow(far, W, HFOV, cam_x, face1) == 1
    assert V.arrow_scale(near, W, HFOV, face1 - cam_x) == pytest.approx(V.ARROW_BOX_SCALE, rel=0.01)


def test_confident_bullseyes_only():
    assert V.confident(['99', 0.8, 0, 0, 20, 20])
    assert not V.confident(['99', 0.65, 0, 0, 20, 20])     # the textures' false matches scored 0.64-0.67
    assert not V.confident(['99', 0.9, 0, 0, 10, 10])      # smaller than the bar can be while measuring


# ---- the bar's width from the gap between arrow 2 and a bullseye (bar_width) ----

ARROW_PX = SYMBOL_M * V.ARROW_BOX_SCALE
RANGE = (0.30, 1.20)


def bar_frame(true_cam, believed_cam, width, face_x, stamp=1.0, ends=(1,), arrow=True):
    """One frame of the bar seen from true_cam, logged with believed_cam (the pose
    the car thinks it has): bullseyes on `ends` (+1 left / -1 right), arrow 2 at the centre."""
    bulls = [(stamp, project(true_cam, (face_x, CY + e * (width / 2.0 - 0.057))), W, believed_cam) for e in ends]
    a = project(true_cam, (face_x, CY), size=ARROW_PX)
    a[0] = '39'
    return bulls, ([(stamp, a)] if arrow else [])


@pytest.mark.parametrize('d2, width, side', [(0.60, 0.40, 1), (1.00, 0.60, -1), (1.50, 0.80, 1), (1.00, 1.00, -1)])
def test_gap_method_ignores_heading_and_position_error(d2, width, side):
    face_x = 1.30 + d2
    true_cam = (1.20, CY + side * 0.25, math.radians(-3.0 * side))   # yawed 3 deg toward the bar's centre
    believed = (1.20, CY + side * 0.28, 0.0)                         # gyro / odometry: straight, 3 cm out
    bulls, arrows = bar_frame(true_cam, believed, width, face_x, ends=(side,))
    got, _, how = V.bar_width(bulls, arrows, HFOV, face_x, CY, side, RANGE)
    assert how == 'gap' and got == pytest.approx(width, abs=0.01)
    old, _, how = V.bar_width(bulls, [], HFOV, face_x, CY, side, RANGE)   # the pose method, same frame
    assert how == 'pose' and abs(old - width) > 0.05


def test_both_ends_in_one_frame_needs_the_arrow():
    # Arrow + one bullseye only (2026-10-15): both ends without arrow 2 are no gap reading.
    face_x, true_cam = 2.30, (1.20, CY + 0.25, math.radians(3.0))
    believed = (1.20, CY + 0.25, 0.0)
    bulls, arrows = bar_frame(true_cam, believed, 0.60, face_x, ends=(1, -1), arrow=True)
    got, widths, how = V.bar_width(bulls, arrows, HFOV, face_x, CY, 1, RANGE)
    assert how == 'gap' and got == pytest.approx(0.60, abs=0.02)
    bulls, arrows = bar_frame(true_cam, believed, 0.60, face_x, ends=(1, -1), arrow=False)
    assert V.bar_width(bulls, arrows, HFOV, face_x, CY, 1, RANGE)[2] != 'gap'


def test_near_end_preferred():
    face_x, cam = 2.30, (1.20, CY + 0.25, 0.0)
    near, arrows = bar_frame(cam, cam, 0.60, face_x, stamp=1.0, ends=(1,))
    far, arrows_far = bar_frame(cam, cam, 0.80, face_x, stamp=2.0, ends=(-1,))   # a wrong far-end reading
    got, widths, _ = V.bar_width(near + far, arrows + arrows_far, HFOV, face_x, CY, 1, RANGE)
    assert widths == [pytest.approx(0.60, abs=0.005)]
    got, _, _ = V.bar_width(far, arrows_far, HFOV, face_x, CY, 1, RANGE)      # only the far end: used
    assert got == pytest.approx(0.80, abs=0.01)


def test_falls_back_to_the_pose_without_arrow_2_in_the_frame():
    face_x, cam = 2.30, (1.20, CY + 0.25, 0.0)
    bulls, arrows = bar_frame(cam, cam, 0.60, face_x, stamp=1.0)
    late = [(1.1, a) for _, a in arrows]                                # another frame's arrow: not paired
    got, _, how = V.bar_width(bulls, late, HFOV, face_x, CY, 1, RANGE)
    assert how == 'pose' and got == pytest.approx(0.60, abs=0.005)
    assert V.bar_width([], arrows, HFOV, face_x, CY, 1, RANGE) == (None, [], 'pose')


def test_gap_refuses_a_wrong_sized_bullseye():
    face_x, cam = 2.30, (1.20, CY + 0.25, 0.0)
    bulls, arrows = bar_frame(cam, cam, 0.60, face_x)
    big = project(cam, (face_x, CY + 0.25), size=BULLSEYE_RIM_M * 3)
    got, _, how = V.bar_width([(1.0, big, W, cam)], arrows, HFOV, face_x, CY, 1, RANGE)
    assert got is None


def test_gap_method_at_the_checkpoint_1_glance():
    # Checkpoint 1, d2 = 0.6 m: turned 15 deg toward the centre line so arrow 2 is in view.
    # The gyro 3 deg off costs ~2 cm (arrow 2 is 32 deg off square to the face), the
    # position 3 cm off nothing; the old ray method is off 11 cm with both.
    face_x = 1.30 + 0.60
    true_cam = (1.42, CY + 0.30, math.radians(-15.0))
    believed = (1.42, CY + 0.33, math.radians(-12.0))
    bulls, arrows = bar_frame(true_cam, believed, 0.60, face_x, ends=(1,))
    got, _, how = V.bar_width(bulls, arrows, HFOV, face_x, CY, 1, RANGE)
    assert how == 'gap' and got == pytest.approx(0.60, abs=0.025)
    old, _, _ = V.bar_width(bulls, [], HFOV, face_x, CY, 1, RANGE)
    assert old is None or abs(old - 0.60) > 0.08


def test_a_bullseye_from_yolo_and_the_matcher_counts_once():
    face_x, cam = 2.30, (1.20, CY + 0.25, 0.0)
    bulls, arrows = bar_frame(cam, cam, 0.60, face_x)
    t, box, w, c = bulls[0]
    yolo = box + ['yolo']
    matcher = [box[0], 0.95, box[2] + 1, box[3] - 1, box[4] + 1, box[5] - 1, 'matcher']
    assert V.merge_duplicates([matcher, yolo]) == [yolo]
    got, widths, how = V.bar_width([(t, matcher, w, c), (t, yolo, w, c)], arrows, HFOV, face_x, CY, 1, RANGE)
    assert how == 'gap' and len(widths) == 1 and got == pytest.approx(0.60, abs=0.01)
