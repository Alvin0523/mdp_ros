"""tasks/drive_past.py: a box is credited to a block only where that block's
face must appear in the image, the right size, consistently."""
import math

from mdp_bringup.tasks.drive_past import Camera, DrivePast, candidates, match

CAM = Camera(0.0, -0.081, 0.12, math.pi / 2)        # looks left, like the car's
BLOCKS = [(1.0, 1.0), (1.4, 1.0)]
FACES = [(0.0, -1.0), (0.0, -1.0)]                   # both images face S


def box_for(pose, i, tid='11', grow=1.0):
    (i_, u, v, w), = [c for c in candidates(CAM, pose, BLOCKS, FACES, [i])]
    h = w * grow
    return (tid, 0.9, u - w * grow / 2, v - h / 2, u + w * grow / 2, v + h / 2)


def pose_at(x):
    return (x, 0.62, 0.0)                            # driving east, 0.30 m south of the faces


def test_face_in_view_only_near_square():
    assert [c[0] for c in candidates(CAM, pose_at(1.0), BLOCKS, FACES, [0, 1])] == [0]
    assert candidates(CAM, pose_at(0.5), BLOCKS, FACES, [0, 1]) == []      # far off to the side


def test_box_matches_its_own_face_not_the_other():
    pose = pose_at(1.0)
    assert match(candidates(CAM, pose, BLOCKS, FACES, [0, 1]), [box_for(pose, 0)], 640, 480) == [(0, '11')]
    # The same box with only the other block unread: not where its face is.
    assert match(candidates(CAM, pose, BLOCKS, FACES, [1]), [box_for(pose, 0)], 640, 480) == []


def test_wrong_size_not_matched():
    pose = pose_at(1.0)
    cands = candidates(CAM, pose, BLOCKS, FACES, [0])
    assert match(cands, [box_for(pose, 0, grow=2.0)], 640, 480) == []


def test_read_after_three_frames_over_five_cm():
    dp = DrivePast(CAM)
    dp.reset(BLOCKS, FACES)
    got = []
    for x in (0.97, 0.99, 1.01, 1.03):
        got += dp.on_frame(pose_at(x), [box_for(pose_at(x), 0, '32')], 640, 480, [0, 1])
    assert got == [(0, '32')]


def test_two_ids_on_one_face_never_read():
    dp = DrivePast(CAM)
    dp.reset(BLOCKS, FACES)
    got = []
    for x, tid in ((0.97, '32'), (0.99, '11'), (1.01, '32'), (1.03, '32'), (1.05, '32')):
        got += dp.on_frame(pose_at(x), [box_for(pose_at(x), 0, tid)], 640, 480, [0, 1])
    assert got == [] and 0 in dp.conflict
