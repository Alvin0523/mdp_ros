"""`pixi run setup` for task 2: tasks.yaml's task2 setup layout survives the trip
over /task2_setup, and lands where the Gazebo arena puts the obstacles."""
from pathlib import Path

import pytest

from mdp_bringup.utils import obstacle_layout

TASKS_YAML = Path(__file__).resolve().parents[1] / 'config' / 'tasks.yaml'


def test_round_trip_matches_the_gazebo_arena():
    x1, x2, a1, a2, width = obstacle_layout.from_task2_setup(obstacle_layout.task2_setup_string(str(TASKS_YAML)))
    _, _, (arena_x1, arena_x2), arena_width = obstacle_layout.task2_setup(str(TASKS_YAML))
    assert x1 == pytest.approx(arena_x1, abs=1e-3)
    assert x2 == pytest.approx(arena_x2, abs=1e-3)
    assert {a1, a2} <= {'LEFT', 'RIGHT'}
    assert width == pytest.approx(arena_width, abs=1e-3)


def test_parses_a_hand_written_message():
    assert obstacle_layout.from_task2_setup('1:0.75,left|2:1.45,RIGHT') == (0.75, 1.45, 'LEFT', 'RIGHT', None)


@pytest.mark.parametrize('bad', ['', '1:0.75,LEFT', '1:0.75,UP|2:1.45,RIGHT', '1:x,LEFT|2:1.45,RIGHT'])
def test_rejects_malformed(bad):
    with pytest.raises((ValueError, KeyError)):
        obstacle_layout.from_task2_setup(bad)
