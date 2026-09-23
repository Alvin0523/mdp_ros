"""Unit tests for scripts/task1_logic.py (pure helpers of task1_runner.py)."""
import importlib.util
import math
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'task1_logic.py'
_spec = importlib.util.spec_from_file_location('task1_logic', _PATH)
logic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logic)


def test_parse_setup_keeps_tablet_ids_and_facing():
    got = logic.parse_setup("3:0.55,1.25,N|1:1.05,0.35,east")
    assert got == [(3, 0.55, 1.25, 'N'), (1, 1.05, 0.35, 'E')]


def test_parse_setup_skips_malformed_items():
    got = logic.parse_setup("junk|2:abc,1,N|4:0.5,0.5,X|5:0.5,0.5,S")
    assert got == [(5, 0.5, 0.5, 'S')]


def test_parse_setup_non_numeric_id_falls_back_to_position():
    assert logic.parse_setup("a:0.5,0.5,N") == [(1, 0.5, 0.5, 'N')]


def test_parse_setup_cells_are_centred_in_the_cell():
    got = logic.parse_setup("1:0,1,N|2:9,8,W|3:19,19,S", cells=True)
    assert [(i, round(x, 3), round(y, 3), f) for i, x, y, f in got] == [
        (1, 0.05, 0.15, 'N'), (2, 0.95, 0.85, 'W'), (3, 1.95, 1.95, 'S')]


def test_parse_setup_cells_rejects_out_of_range_and_fractions():
    assert logic.parse_setup("1:20,0,N|2:-1,0,N|3:1.5,0,N|4:2,2,N", cells=True) == [
        (4, 0.25, 0.25, 'N')]


def test_pose_at_start_position_and_heading():
    start = (0.15, 0.15, math.pi / 2)
    tol = math.radians(10)
    assert logic.pose_at_start((0.17, 0.14, math.pi / 2 + 0.05), start, 0.05, tol)
    assert not logic.pose_at_start((0.30, 0.15, math.pi / 2), start, 0.05, tol)
    assert not logic.pose_at_start((0.15, 0.15, 0.0), start, 0.05, tol)


def test_pose_at_start_wraps_yaw():
    start = (0.0, 0.0, math.pi)
    assert logic.pose_at_start((0.0, 0.0, -math.pi + 0.02), start, 0.05, 0.1)


def test_indicators_idle_not_ready():
    lines = logic.indicator_lines('WAITING_FOR_SETUP', False, False, False, True, None)
    assert lines == ['PLAN:WAITING', 'RESET:DONE', 'ESTOP:OFF', 'STATUS:Not ready']


def test_indicators_ready_needs_both():
    ready = logic.indicator_lines('WAITING_FOR_GO', True, False, True, True, None)
    assert ready == ['PLAN:DONE', 'RESET:DONE', 'ESTOP:OFF', 'STATUS:Ready']
    unreset = logic.indicator_lines('WAITING_FOR_GO', True, False, True, False, None)
    assert unreset == ['PLAN:DONE', 'RESET:WAITING', 'ESTOP:OFF', 'STATUS:Not ready']


def test_indicators_planning_in_progress():
    lines = logic.indicator_lines('WAITING_FOR_GO', True, True, False, True, None)
    assert lines[0] == 'PLAN:PLANNING'


def test_indicators_run_reports_status_only():
    assert logic.indicator_lines('NAVIGATING_TO_TARGET', True, False, True, False, 3) == \
        ['ESTOP:OFF', 'STATUS:Going to obstacle 3']
    assert logic.indicator_lines('PAUSE_FOR_SCAN', True, False, True, False, 3) == \
        ['ESTOP:OFF', 'STATUS:Scanning obstacle 3']


def test_indicators_finished_and_stopped_keep_plan_done():
    assert logic.indicator_lines('FINISHED', True, False, True, False, None) == \
        ['PLAN:DONE', 'RESET:WAITING', 'ESTOP:OFF', 'STATUS:Finished']
    assert logic.indicator_lines('STOPPED', True, False, True, False, None)[3] == 'STATUS:Stopped'


def test_estop_blocks_ready_and_is_always_reported():
    idle = logic.indicator_lines('WAITING_FOR_GO', True, False, True, True, None, estop=True)
    assert idle == ['PLAN:DONE', 'RESET:DONE', 'ESTOP:ON', 'STATUS:Not ready']
    run = logic.indicator_lines('NAVIGATING_TO_TARGET', True, False, True, False, 2, estop=True)
    assert run == ['ESTOP:ON', 'STATUS:Going to obstacle 2']
