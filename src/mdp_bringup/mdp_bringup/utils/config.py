"""mdp_bringup/config/navigation.yaml as defaults for a node's parameters.

The launch passes the same file as the node's parameters, so these are only
used by a bare `ros2 run` - but they mean the numbers live in the YAML alone.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory


def navigation(section: str) -> dict:
    """ros__parameters of one section ('task2_runner', '/**') of navigation.yaml."""
    path = os.path.join(get_package_share_directory('mdp_bringup'), 'config', 'navigation.yaml')
    with open(path) as f:
        return yaml.safe_load(f)[section]['ros__parameters']


def declare(node, section: str) -> None:
    """Declare every parameter of `section`, defaulting to the YAML's value."""
    for name, value in navigation(section).items():
        node.declare_parameter(name, value)
