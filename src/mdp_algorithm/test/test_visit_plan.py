"""visit_plan.plan_visits() (visit.mode anchors) on the tasks.yaml task 1 layout and
the 2026-10-14 tablet layout: every image is visited or driven by exactly once, the
stops and vias are clear of the blocks, each drive-by is in view on its leg, and it
plans fast (it runs on the Pi before START)."""
import math
import time

import pytest
import yaml

from mdp_algorithm.planning import visit_plan
from mdp_algorithm.planning.planner import CM_PER_M
from mdp_algorithm.utils import params

START = (0.15, 0.15, math.pi / 2)        # cell (1,1) facing N
CAMERA = (0.0, -0.055, math.pi / 2)      # URDF, task 1: beside the rear axle, looking left


def _tasks_yaml():
    with open(params._share('mdp_bringup', 'config', 'tasks.yaml')) as f:
        entries = yaml.safe_load(f)['task1']['obstacles']
    return [(e['cell_x'] * 10.0 + 5.0, e['cell_y'] * 10.0 + 5.0, e['facing']) for e in entries]


# The layout on the tablet 2026-10-14 (read off a photo), cells + image side.
TABLET = [(c * 10.0 + 5.0, r * 10.0 + 5.0, f) for c, r, f in
          [(1, 17, 'S'), (15, 18, 'S'), (6, 12, 'W'), (12, 14, 'S'),
           (18, 11, 'W'), (12, 8, 'E'), (7, 5, 'N'), (14, 3, 'E')]]


@pytest.mark.parametrize('layout', [_tasks_yaml(), TABLET], ids=['tasks.yaml', 'tablet'])
def test_plan_visits(layout):
    t0 = time.time()
    visits, unreachable, costmap = visit_plan.plan_visits(layout, START, CAMERA)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f'{elapsed:.1f} s'

    stopped = [v.obstacle for v in visits]
    driven = [o for v in visits for o in v.drive_bys]
    assert sorted(stopped + driven + unreachable) == list(range(len(layout)))
    assert all(v.kind in ('anchor', 'angled') for v in visits[:-1]) and visits[-1].kind == 'last'
    assert all(len(v.drive_bys) <= params.ACTIVE.max_drive_bys_per_leg for v in visits)

    for v in visits:
        for x, y, th in [v.pose] + v.vias:
            assert not costmap.in_collision(x * CM_PER_M, y * CM_PER_M, th)

    # every drive-by in view on its estimated leg (through its via)
    pl = visit_plan._Planner(costmap, (START[0] * CM_PER_M, START[1] * CM_PER_M, START[2]),
                             (CAMERA[0] * CM_PER_M, CAMERA[1] * CM_PER_M, CAMERA[2]), 0.3)
    cm = lambda p: (p[0] * CM_PER_M, p[1] * CM_PER_M, p[2])
    prev = pl.start
    for v in visits:
        points = [prev] + [cm(q) for q in v.vias] + [cm(v.pose)]
        for o in v.drive_bys:
            run = max(pl.view_run(pl.rs(a, b)[1], o) for a, b in zip(points, points[1:]))
            assert run >= params.ACTIVE.drive_by_min_view * CM_PER_M, f'drive-by {o}: {run:.0f} cm'
        prev = points[-1]


def test_every_image_on_tasks_yaml_visited():
    """#6 and #8 face each other 20 cm apart: no square stop fits - angled stops (or drive-bys)."""
    _, unreachable, _ = visit_plan.plan_visits(_tasks_yaml(), START, CAMERA)
    assert not unreachable


def test_fewer_stops_on_the_tablet_layout():
    visits, unreachable, _ = visit_plan.plan_visits(TABLET, START, CAMERA)
    assert not unreachable
    assert len(visits) < len(TABLET)
