"""mdp.launch.py builds for sim and real, every task and role, and every config
file it names exists - catches typos and renamed files before a launch on the car."""
import importlib.util
import re
from pathlib import Path

import pytest
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch_ros.actions import Node

PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / 'launch' / 'mdp.launch.py'


def _launch_module():
    spec = importlib.util.spec_from_file_location('mdp_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_gazebo() -> bool:
    try:
        get_package_share_directory('ros_gz_sim')
        return True
    except PackageNotFoundError:
        return False   # the Pi: Gazebo is laptop-only (pixi.toml)


def _executables(args):
    description = _launch_module().generate_launch_description(args + ['gui:=false'])
    return {e.node_executable for e in description.entities if isinstance(e, Node)}


@pytest.mark.parametrize('sim', [
    pytest.param('true', marks=pytest.mark.skipif(not _has_gazebo(), reason='no Gazebo here')),
    'false'])
@pytest.mark.parametrize('task', ['0', '1', '2'])
def test_launch_builds(sim, task):
    description = _launch_module().generate_launch_description(
        [f'sim:={sim}', f'task:={task}', 'vision:=true', 'gui:=false'])
    assert description.entities


MONITORS = {'health_monitor', 'bt_monitor'}


@pytest.mark.parametrize('task', ['0', '1', '2'])
def test_split_roles(task):
    """Pi + laptop: the Pi side has the hardware and the runner but not the
    monitors; the laptop side has the planner and the monitors, no hardware."""
    solo = _executables(['sim:=false', f'task:={task}'])
    pi = _executables(['sim:=false', f'task:={task}', 'role:=pi'])
    laptop = _executables(['sim:=false', 'role:=laptop'])
    assert MONITORS <= solo and not MONITORS & pi
    assert MONITORS | {'task1_planner'} <= laptop
    assert not {'serial_bridge_node', 'bluetooth_bridge_node', 'ros2_control_node'} & laptop
    assert pi | laptop >= solo   # nothing lost in the split


def test_config_files_named_in_the_launch_exist():
    text = LAUNCH.read_text() + (PACKAGE / 'launch' / 'vision.launch.py').read_text()
    names = set(re.findall(r"'([\w]+\.yaml)'", text))
    assert names, 'no config files found in the launch files'
    missing = [n for n in names if not (PACKAGE / 'config' / n).is_file()]
    assert not missing, missing


def test_every_executable_in_the_launch_is_installed():
    """Each mdp_bringup executable the launch starts has an entry point in setup.py."""
    used = set(re.findall(r"package='mdp_bringup', executable='(\w+)'", LAUNCH.read_text()))
    setup = (PACKAGE / 'setup.py').read_text()
    missing = [n for n in used if f"'{n}'" not in setup]
    assert used and not missing, missing
