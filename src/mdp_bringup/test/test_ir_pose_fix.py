"""tasks/ir_pose_fix.py: a car whose pose estimate is off by a known amount drives
past a block with two left IRs; the fix must find that amount."""
import math

from mdp_bringup.tasks.ir_pose_fix import BLOCK_HALF, Face, IrPoseFix

SENSORS = {'ir': (0.04, 0.085, math.pi / 2), 'ir2': (-0.04, 0.085, math.pi / 2)}
BLOCK = (1.0, 1.0)            # block centre; image side faces S (-y)
FACE = Face(*BLOCK, 0.0, -1.0)
CAR_Y = 1.0 - 0.30            # base_link 30 cm south of the block
YAW = 0.0                     # driving east, left (camera, IRs) = north


def true_reading(sensor, pose):
    """What the IR really reads: the face when its beam hits it, else far."""
    beam = FACE.beam(sensor, pose)
    if beam is not None and abs(beam[1]) <= BLOCK_HALF:
        return beam[0]
    return 0.80


def readings(fix, x, error):
    ex, ey = error
    for name, sensor in SENSORS.items():
        fix.on_reading(name, true_reading(sensor, (x, CAR_Y, YAW)), (x + ex, CAR_Y + ey, YAW))


def drive(fix, error, stop_x):
    """Drive east along y = CAR_Y and stop with base_link at stop_x; then, as
    the runner does, creep towards the IR that sees the face until both do.
    The estimate is the truth + error."""
    x = 0.60
    while x < stop_x:
        x = min(stop_x, x + 0.006)                       # 0.15 m/s at 25 Hz
        readings(fix, x, error)
    fix.at_stop()
    readings(fix, x, error)
    seen = fix.seen()
    if seen in ('front', 'rear'):
        step = 0.002 if seen == 'front' else -0.002      # 5 cm/s at 25 Hz
        while fix.seen() != 'both':
            x += step
            readings(fix, x, error)
        fix.at_stop()
    for _ in range(5):
        readings(fix, x, error)
    return x


def test_finds_the_error():
    for error in [(0.0, 0.0), (0.03, 0.0), (-0.025, 0.02), (0.02, -0.03)]:
        for stop_off in (-0.03, 0.03):
            fix = IrPoseFix(SENSORS)
            fix.reset(FACE)
            # Stopped 3 cm off the checkpoint (base_link square to the block centre).
            x = drive(fix, error, stop_x=BLOCK[0] + stop_off)
            dx, dy, note = fix.correction()
            assert abs(dx + error[0]) < 0.005, (error, stop_off, note)
            assert abs(dy + error[1]) < 0.003, (error, stop_off, note)
            # Then the runner centres the car: from the corrected pose.
            along, per_m = fix.along_offset((x + error[0] + dx, CAR_Y + error[1] + dy, YAW))
            assert abs(along - (x - BLOCK[0])) < 0.005


def test_both_on_face_keeps_beams_on_it():
    """No edge, both IRs on the face, but the estimate puts the front beam 2 cm
    past the edge: moved back just enough."""
    fix = IrPoseFix(SENSORS)
    fix.reset(FACE)
    fix.at_stop()
    est = (BLOCK[0] + 0.03, CAR_Y, YAW)          # truth: square to the block
    for _ in range(5):
        for name, sensor in SENSORS.items():
            fix.on_reading(name, true_reading(sensor, (BLOCK[0], CAR_Y, YAW)), est)
    dx, _, note = fix.correction()
    assert abs(dx + 0.02) < 0.002, note


def test_no_face_no_fix():
    fix = IrPoseFix(SENSORS)
    fix.reset(FACE)
    fix.at_stop()
    for _ in range(5):
        fix.on_reading('ir', 0.80, (1.0, CAR_Y, YAW))
    assert fix.correction()[:2] == (0.0, 0.0)


def test_misreading_not_used():
    """The two IRs disagree by 5 cm on the gap (expected 16.5 cm): too unsure,
    nothing moves."""
    fix = IrPoseFix(SENSORS)
    fix.reset(FACE)
    fix.at_stop()
    fix.on_reading('ir', 0.19, (1.0, CAR_Y, YAW))
    fix.on_reading('ir2', 0.14, (1.0, CAR_Y, YAW))
    assert fix.correction()[:2] == (0.0, 0.0)


def test_seen_says_which_way_to_creep():
    """Stopped 3 cm short of the block centre (driving east): only the front IR
    is on the face - creep forward. 3 cm past it: only the rear - creep back."""
    for stop_x, want in [(BLOCK[0] - 0.03, 'front'), (BLOCK[0] + 0.03, 'rear'), (BLOCK[0], 'both'),
                         (BLOCK[0] + 0.12, 'none')]:
        fix = IrPoseFix(SENSORS)
        fix.reset(FACE)
        for name, sensor in SENSORS.items():
            fix.on_reading(name, true_reading(sensor, (stop_x, CAR_Y, YAW)), (stop_x, CAR_Y, YAW))
        assert fix.seen() == want, (stop_x, fix.seen())
