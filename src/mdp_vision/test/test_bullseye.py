"""bullseye.py: the printed bullseye pasted into a grey frame at a few sizes is
found where it was put, at its size; a frame without one gives nothing."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from mdp_vision.bullseye import find_bullseyes, load_template

TEMPLATE = Path(__file__).resolve().parents[2] / 'mdp_description' / 'models' / 'symbols' / '99_Bullseye.png'


@pytest.fixture(scope='module')
def template():
    return load_template(str(TEMPLATE))


def frame_with(template, boxes):
    rng = np.random.default_rng(1)
    frame = (rng.normal(110, 12, (480, 640))).clip(0, 255).astype(np.uint8)
    for x, y, s in boxes:
        frame[y:y + s, x:x + s] = cv2.resize(template, (s, s), interpolation=cv2.INTER_AREA)
    return frame


@pytest.mark.parametrize('size', [16, 22, 40, 70])
def test_finds_one_at_its_size(template, size):
    x, y = 300, 200
    found = find_bullseyes(frame_with(template, [(x, y, size)]), template, min_px=14.0)   # vision.yaml
    assert found
    _, x1, y1, x2, y2 = found[0]
    assert abs(x1 - x) <= 0.1 * size + 1 and abs(y1 - y) <= 0.1 * size + 1
    assert abs((x2 - x1) - size) <= 0.04 * size + 1          # the runner reads distance from it


def test_two_far_apart(template):
    found = find_bullseyes(frame_with(template, [(220, 222, 22), (400, 222, 22)]), template, min_px=14.0)
    xs = sorted(round(b[1]) for b in found[:2])
    assert len(found) >= 2 and abs(xs[0] - 220) <= 4 and abs(xs[1] - 400) <= 4


def test_nothing_in_an_empty_frame(template):
    assert find_bullseyes(frame_with(template, []), template, min_px=14.0) == []
