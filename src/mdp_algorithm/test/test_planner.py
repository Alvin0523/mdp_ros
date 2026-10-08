"""Planner checks on the Task 1 layout in mdp_bringup/config/tasks.yaml.

Runs the same calls task1_runner makes - plan_visiting_order(), then plan_leg()
for every leg - and checks every obstacle is reachable, every leg is found, and
no pose on any path collides. Plus the costmap values and the angle helper.
"""
import math

import yaml

from mdp_algorithm.planning.costmap import INSCRIBED, LETHAL, Costmap, Obstacle
from mdp_algorithm.planning.planner import plan_leg, plan_visiting_order
from mdp_algorithm.utils.geometry import M
from mdp_algorithm.utils import params



def _task1_layout():
    """tasks.yaml task1: block centres in cm (cell centre) and the image face."""
    with open(params._share('mdp_bringup', 'config', 'tasks.yaml')) as f:
        entries = yaml.safe_load(f)['task1']['obstacles']
    return [(e['cell_x'] * 10.0 + 5.0, e['cell_y'] * 10.0 + 5.0, e['facing']) for e in entries]


LAYOUT = _task1_layout()
START = (0.15, 0.15, math.pi / 2)        # cell (1,1) facing N
CAMERA_LEFT = math.pi / 2


def test_every_obstacle_is_reached_without_collision():
    """Every obstacle with a stop is driven to without collision. A block can lose
    its stop - with 6 cm block padding (2026-10-08) the tight layout's #6 and #8 do,
    their slid stops would need over 10 cm of slide - but never more than two, and
    every obstacle is either visited or reported unreachable."""
    order, checkpoints, unreachable, costmap = plan_visiting_order(LAYOUT, START, CAMERA_LEFT)
    assert len(unreachable) <= 2, f'unreachable: {unreachable}'
    assert sorted(list(order) + list(unreachable)) == list(range(len(LAYOUT)))

    goal_tol = params.ACTIVE.goal_xy_tolerance
    pose = START
    for checkpoint in checkpoints:
        leg = plan_leg(costmap, pose, checkpoint)
        assert leg, f'no path to {checkpoint}'
        for x, y, theta, _gear in leg:
            assert not costmap.in_collision(x * 100.0, y * 100.0, theta)
        end = leg[-1]
        assert math.hypot(end[0] - checkpoint[0], end[1] - checkpoint[1]) <= goal_tol + 1e-9
        pose = checkpoint


def test_costmap_values():
    costmap = Costmap([Obstacle(105.0, 105.0, 'N', 0)])
    assert costmap.cost_at(105.0, 105.0) == LETHAL              # inside the block
    assert costmap.cost_at(111.0, 105.0) == INSCRIBED           # 1 cm from its edge
    assert costmap.cost_at(150.0, 150.0) == 0                   # far from everything
    assert costmap.cost_at(-1.0, 50.0) == LETHAL                # off the table
    assert costmap.in_collision(95.0, 105.0, 0.0)               # car nose into the block
    assert not costmap.in_collision(60.0, 60.0, 0.0)


def test_yaw_wrap():
    assert M(0.0) == 0.0
    assert math.isclose(M(math.pi), -math.pi, abs_tol=1e-9)
    assert math.isclose(M(3 * math.pi / 2), -math.pi / 2, abs_tol=1e-9)
    for k in range(-20, 21):
        assert -math.pi - 1e-9 <= M(k * 0.7) < math.pi + 1e-9
