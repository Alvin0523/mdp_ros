"""mdp.launch.py builds for sim and real, every task, and every config file it
names exists - catches typos and renamed files before a launch on the car."""
import importlib.util
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / 'launch' / 'mdp.launch.py'


def _launch_module():
    spec = importlib.util.spec_from_file_location('mdp_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('sim', ['true', 'false'])
@pytest.mark.parametrize('task', ['0', '1', '2'])
def test_launch_builds(sim, task):
    description = _launch_module().generate_launch_description(
        [f'sim:={sim}', f'task:={task}', 'vision:=true', 'gui:=false'])
    assert description.entities


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
