"""Task 2's checkpoint 1 -> 2 route table (tasks/task2_routes.py): the shipped
table matches the settings in config/, covers every layout, its routes are
clear of the obstacles, and a lookup ends exactly at checkpoint 2."""
import dataclasses
import math
from pathlib import Path

import pytest

from mdp_algorithm.planning import task2_paths
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tasks import task2_routes as R
from mdp_bringup.utils import obstacle_layout

CONFIG = Path(__file__).resolve().parents[1] / 'config'
SETTINGS = R.task2_settings(str(CONFIG / 'navigation.yaml'))
ARENA = obstacle_layout.load_task2(str(CONFIG / 'tasks.yaml'))
CAR = dataclasses.replace(planner_params.load(navigation_yaml=str(CONFIG / 'navigation.yaml')),
                          footprint_padding=float(SETTINGS['obstacle_padding']))


@pytest.fixture(scope='module')
def table():
    planner_params.configure(CAR)
    t = R.RouteTable.load(str(CONFIG / R.TABLE_FILE))
    assert t is not None, 'config/task2_routes.json.gz missing: pixi run task2-table'
    return t


def test_shipped_table_matches_the_settings(table):
    # A failure here: a setting the routes depend on changed - rebuild (pixi run task2-table).
    assert table.matches(R.fingerprint(CAR, ARENA, SETTINGS))


def test_every_layout_has_a_route(table):
    for key, rows in table.data['routes'].items():
        for d2, row in zip(table.d2s, rows):
            for w, route in zip(table.widths, row):
                assert route is not None, f'{key} d2 {d2} bar {w}'


@pytest.mark.parametrize('arrows', ['LEFT/LEFT', 'LEFT/RIGHT', 'RIGHT/LEFT', 'RIGHT/RIGHT'])
@pytest.mark.parametrize('d2, width', [(0.62, 0.60), (0.98, 0.40), (1.48, 1.20), (0.75, 0.85)])
def test_lookup_ends_at_checkpoint_2_and_is_clear(table, arrows, d2, width):
    a1, a2 = arrows.split('/')
    width = table.round_width(width)
    local, why = table.lookup(d2, width, a1, a2)
    assert local, why
    x1 = ARENA.opening_x + 0.9 + ARENA.obstacle_1_size[0] / 2.0
    x2 = R.bar_x(ARENA, x1, d2)
    cp1 = R.checkpoint_1(x1, a1, SETTINGS['lane_offset'])
    cp2 = R.bar_checkpoints(x2, a2, width, SETTINGS['side_clearance'])[0]
    path = task2_paths.to_world(cp1, local)
    assert path[-1][:2] == pytest.approx(cp2[:2], abs=0.005)
    assert abs(math.remainder(path[-1][2] - cp2[2], 2 * math.pi)) < 0.02
    planner_params.configure(CAR)
    cm = R.course_costmap(ARENA, x1, x2, width, SETTINGS['obstacle_padding'], SETTINGS['wall_padding'])
    assert task2_paths.clear(path, cm, margin=0.0)


def test_width_rounds_to_the_nearest(table):
    assert table.round_width(0.401) == pytest.approx(0.40)
    assert table.round_width(0.43) == pytest.approx(0.45)
    assert table.round_width(1.3) is None


def test_lookup_refuses_outside_the_table(table):
    assert table.lookup(2.0, 0.6, 'LEFT', 'LEFT')[0] is None
    assert table.lookup(0.50, 0.6, 'LEFT', 'LEFT')[0] is None
    assert table.lookup(0.57, 0.6, 'LEFT', 'LEFT')[0] is not None   # a few cm short of the course's 0.60
    assert table.lookup(1.56, 0.6, 'LEFT', 'LEFT')[0] is not None   # the ultrasonic's long reading at 1.5 m
    assert table.lookup(1.0, 0.62, 'LEFT', 'LEFT')[0] is None      # not a table width


def test_stretch_moves_only_the_final_straight():
    route = [(0.02 * i, 0.0 if i < 20 else min(0.3, 0.03 * (i - 20)), 0.0, 1) for i in range(30)]
    route += [(0.6 + 0.02 * k, 0.3, 0.0, 1) for k in range(1, 16)]
    out = R.stretch(route, 0.02)
    assert out[-1][0] == pytest.approx(route[-1][0] + 0.02)
    assert out[:29] == route[:29]


def test_course_survives_json_to_the_laptop():
    import json
    assert R.arena_from_dict(json.loads(json.dumps(R.arena_to_dict(ARENA)))) == ARENA
