"""Task 2's clean shapes (planning/task2_paths.py) and the follower's forward-only
mode: the smooth S starts and ends pointing straight with no sharper turn than
the planned circles, a found path is clear of the obstacles, and a forward-only
path is never reversed on by the follower itself."""
import math

import pytest

from mdp_algorithm.control.path_follower import PathFollower
from mdp_algorithm.planning import task2_paths as T
from mdp_algorithm.planning.costmap import Costmap, Obstacle

SHAPES = T.Shapes(r_left=0.23, r_right=0.334, straight_speed=0.9, max_lateral_accel=1.0)


def empty_course():
    return Costmap([Obstacle(400.0, 120.0, 'W', 0)], arena_cm=(500.0, 240.0))


@pytest.mark.parametrize('length, offset', [(1.0, 0.25), (1.0, -0.25), (1.5, 0.6), (2.0, -0.8)])
def test_quintic_s_is_straight_at_both_ends(length, offset):
    s = T.quintic_s(length, offset)
    assert s[-1][:3] == pytest.approx((length, offset, 0.0))
    assert abs(s[0][2]) < 0.02 and abs(s[-2][2]) < 0.02          # heading ~0 at both ends
    k = T.curvatures(s)
    peak = max(abs(v) for v in k)
    # The curvature changes gradually everywhere - from ~0 at the ends - so the
    # steering never has to jump (the arc S jumps from 0 to full curvature at once).
    assert abs(k[0]) < 0.35 * peak and abs(k[-1]) < 0.35 * peak
    assert max(abs(b - a) for a, b in zip(k, k[1:])) < 0.25 * peak
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(s, s[1:])]
    assert max(steps) < 0.025


def test_arc_s_jumps_where_the_smooth_s_does_not():
    cm = Costmap([], arena_cm=(500.0, 240.0))
    arc = T.arc_s_path((0.7, 1.2, 0.0), (1.6, 1.45, 0.0), cm, SHAPES)
    k = T.curvatures(arc)
    assert max(abs(b - a) for a, b in zip(k, k[1:])) > 0.5 * max(abs(v) for v in k)


def test_turning_circle_check():
    assert not T.s_fits_turning_circles(0.3, 0.5, SHAPES)       # half a metre across in 30 cm: too sharp
    assert T.s_fits_turning_circles(2.0, 0.25, SHAPES)


@pytest.mark.parametrize('dy', [0.25, -0.25, 0.5])
def test_smooth_s_path_reaches_the_goal_within_the_circles(dy):
    cm = empty_course()
    start, goal = (0.7, 1.2, 0.0), (2.0, 1.2 + dy, 0.0)
    path = T.smooth_s_path(start, goal, cm, SHAPES)
    assert path is not None
    assert path[-1][:2] == pytest.approx(goal[:2], abs=1e-6)
    assert abs(path[-1][2]) < 1e-6
    for k in T.curvatures(path):
        assert k <= 1.0 / SHAPES.r_left * 1.15 and -k <= 1.0 / SHAPES.r_right * 1.15
    assert all(p[3] == 1 for p in path)
    assert T.clear(path, cm)


def test_lined_up_path_ends_with_a_straight():
    cm = empty_course()
    goal = (2.0, 1.45, 0.0)
    path = T.stepped_lined_up_path((0.7, 1.2, 0.0), goal, cm, SHAPES, 0.30, 0.04)
    tail = [p for p in path if p[0] >= goal[0] - 0.29]
    assert tail and all(abs(p[1] - goal[1]) < 1e-6 and abs(p[2]) < 1e-6 for p in tail)


def test_a_blocked_s_is_not_returned():
    # A wall of blocks across the whole course between start and goal.
    cm = Costmap([], arena_cm=(500.0, 240.0), walls=[(150.0, 0.0, 160.0, 240.0)])
    assert T.smooth_s_path((0.7, 1.2, 0.0), (2.5, 1.45, 0.0), cm, SHAPES) is None
    assert T.arc_s_path((0.7, 1.2, 0.0), (2.5, 1.45, 0.0), cm, SHAPES) is None


def test_round_path_needs_room_to_turn():
    cm = Costmap([], arena_cm=(500.0, 240.0))
    a, b = (3.0, 1.75, 0.0), (3.0, 0.65, math.pi)
    path = T.round_path(a, b, cm, SHAPES)
    assert path[-1][:2] == pytest.approx(b[:2])
    assert T.round_path((3.0, 1.3, 0.0), (3.0, 1.1, math.pi), cm, SHAPES) is None   # tighter than the car turns


def _follower(path):
    f = PathFollower(wheelbase=0.143, lookahead_dist=0.1, max_steering_left=0.75, max_steering_right=0.57,
                     target_speed=0.6, goal_tolerance=0.035, cusp_tolerance=0.02)
    f.regulate = f.velocity_scaled_lookahead = False
    f.tracking = 'pure_pursuit'
    f.pose_latency = 0.0
    f.set_path(path)
    return f


def test_forward_only_never_reverses_onto_a_point_behind():
    # The car has rolled 0.3 m past the start of a straight path: the target point
    # is behind it. Task 1's rule reverses to it; forward-only turns full lock and
    # keeps going forward instead (task 2 backed up after checkpoint 1, 2026-10-14).
    path = [(1.0 + 0.02 * i, 1.2, 0.0, 1) for i in range(30)]
    task1 = _follower(path)
    task1.update_pose(1.3, 1.19, 0.0)
    assert task1.compute_cmd()[0] < 0.0
    task2 = _follower(path)
    task2.forward_only, task2.behind_speed = True, 0.4
    task2.update_pose(1.3, 1.19, 0.0)
    v, w = task2.compute_cmd()
    assert v == pytest.approx(0.4)
    assert w > 0.0                                # toward the target's side (left: it is at y 1.2)


@pytest.mark.parametrize('across', [0.8, 1.1, 1.45])
def test_round_path_ramps_its_curvature(across):
    cm = Costmap([], arena_cm=(500.0, 240.0))
    a, b = (3.0, 1.2 - across / 2, 0.0), (3.0, 1.2 + across / 2, math.pi)
    path = T.round_path(a, b, cm, SHAPES)
    assert path[-1][:2] == pytest.approx(b[:2]) and abs(path[-2][1] - b[1]) < 0.005   # level with b at the end
    k = T.curvatures(path)
    peak = max(abs(v) for v in k)
    assert abs(k[0]) < 0.25 * peak and peak <= 1.0 / SHAPES.r_left * 1.02
