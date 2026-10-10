"""ir_face_pass: a car driving past a face with its pose off by a known amount
gets that amount back as the correction."""
import math

from mdp_bringup.tasks.ir_face_pass import FacePass

RIGHT_IR = (0.143, -0.062, -math.pi / 2)   # task 2: front axle, looking right


def drive_past(err_x, err_y, speed=0.5, rate=25.0, heading=0.0):
    # Obstacle 1 (10 cm block at x 1.0..1.1, y 1.15..1.25), passed on its left: its +y face.
    fp = FacePass('test', 1.0, 1.1, 1.25, +1.0, RIGHT_IR)
    true_y = 1.25 + 0.138 + 0.062               # IR 13.8 cm out from the face
    step = speed / rate
    x = 0.6
    while x < 1.4:
        truth = (x, true_y, heading)
        bm = fp.beam(truth)
        on = bm is not None and 1.0 <= bm[1] <= 1.1
        reading = bm[0] if on else 0.8          # past the block: far away
        fp.on_reading(reading, (x + err_x, true_y + err_y, heading), speed)
        x += step
    return fp.correction()


def test_no_error_no_fix():
    dx, dy, _ = drive_past(0.0, 0.0)
    assert abs(dx) < 0.012 and abs(dy) < 0.002


def test_recovers_offsets():
    dx, dy, note = drive_past(0.04, -0.03)
    assert abs(dx + 0.04) < 0.012, note        # half a reading step at 25 Hz / 0.5 m/s
    assert abs(dy - 0.03) < 0.003, note


def test_too_big_refused():
    dx, dy, note = drive_past(0.0, 0.07, speed=0.3)
    assert (dx, dy) == (0.0, 0.0) or abs(dy + 0.07) < 0.003, note


def test_out_fix_survives_along_error():
    # +5 cm along left only 2 readings inside the corners by the pose: 'out' was refused.
    dx, dy, note = drive_past(0.05, -0.04)
    assert abs(dx + 0.05) < 0.012, note
    assert abs(dy - 0.04) < 0.003, note
