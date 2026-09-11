"""Shared launch-description and source-introspection helpers.

Every test in this directory asks a question that is settled before a single
message flows - who broadcasts which TF edge, which `frame_id` a publisher
stamps, which start pose the sim spawns at - so all of them read the launch
graph and the runner's AST instead of standing up Gazebo. This module holds the
parts they share; `test_arena_frame_agreement.py` and
`test_path_tracking_frame_consistency.py` grew their own copies first and now
import from here.

Nothing in this module starts a process. `get_launch_description_from_python_
launch_file` executes `generate_launch_description()` only - it never *visits*
the description, so `IncludeLaunchDescription` stays lazy and `gz_sim.launch.py`
is never even read.

Launch files and configs are always read from the SOURCE tree, never from
`install/mdp_bringup/share`, so a stale install copy cannot make a test lie.
"""

import ast
from pathlib import Path

import yaml

from launch import LaunchDescription
from launch_ros.actions import Node as LaunchNode

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_DIR = PACKAGE_ROOT / 'launch'
CONFIG_DIR = PACKAGE_ROOT / 'config'
SCRIPTS_DIR = PACKAGE_ROOT / 'scripts'

# The arena frame the fix introduces (design.md Decision 1). Absent pre-fix.
ARENA_FRAME = 'map'

# The reported case, per bugfix.md: robot spawned inside the start box facing
# arena +Y ("North"), the orientation the planner assumes.
REPORTED_START_POSE = (0.15, 0.15, 1.5707963267948966)

ARENA_SIZE_M = 2.0


# --------------------------------------------------------------------------
# Launch-description introspection
# --------------------------------------------------------------------------

def load_launch(name: str) -> LaunchDescription:
    """Build a launch description from the source tree without visiting it."""
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    return get_launch_description_from_python_launch_file(str(LAUNCH_DIR / name))


def launch_nodes(ld: LaunchDescription):
    return [a for a in ld.entities if isinstance(a, LaunchNode)]


def node_executable(node: LaunchNode) -> str:
    raw = getattr(node, '_Node__node_executable', None) or getattr(node, 'node_executable', '')
    try:
        return ''.join(part.text for part in raw)  # TextSubstitution list
    except (TypeError, AttributeError):
        return str(raw)


def node_executables(ld: LaunchDescription):
    """Sorted executables in the graph - the useful half of a failure message."""
    return sorted(node_executable(n) for n in launch_nodes(ld))


def node_arguments(node: LaunchNode):
    """The node's `arguments` as plain strings, skipping unresolved substitutions."""
    out = []
    for group in getattr(node, '_Node__arguments', None) or []:
        if isinstance(group, str):
            out.append(group)
        elif isinstance(group, (list, tuple)):
            out.extend(str(item) for item in group if isinstance(item, str))
    return out


def node_parameter_dicts(node: LaunchNode):
    """Only the inline `{...}` parameter dicts - file paths are handled separately."""
    return [p for p in (getattr(node, '_Node__parameters', None) or [])
            if isinstance(p, dict)]


def node_parameter_files(node: LaunchNode):
    """Paths of YAML parameter files the node loads, as `Path`s.

    `launch_ros` wraps a path handed to `parameters=[...]` in a `ParameterFile`,
    and normalises the path itself into a list of `TextSubstitution`s that is only
    joined when evaluated against a launch context - which never happens here.
    Hence the three shapes.
    """
    out = []
    for params in getattr(node, '_Node__parameters', None) or []:
        if isinstance(params, (str, Path)):
            out.append(Path(str(params)))
            continue
        raw = getattr(params, 'param_file', None)
        if isinstance(raw, (str, Path)):
            out.append(Path(str(raw)))
        elif isinstance(raw, (list, tuple)):
            text = ''.join(getattr(part, 'text', '') for part in raw)
            if text:
                out.append(Path(text))
    return out


def spawn_start_pose(ld: LaunchDescription):
    """The `(x, y, yaw)` the sim actually spawns the robot at, read off `create`.

    One of the three duplicated start-pose literals root cause 5 calls out (the
    others are `task1_runner`'s planning `start_pose` and the launch docstring),
    so reading it here also pins which literal a render is compared against.
    """
    for node in launch_nodes(ld):
        if node_executable(node) != 'create':
            continue
        args = node_arguments(node)
        pose = {}
        for flag, key in (('-x', 'x'), ('-y', 'y'), ('-Y', 'yaw')):
            if flag in args:
                pose[key] = float(args[args.index(flag) + 1])
        if {'x', 'y', 'yaw'} <= pose.keys():
            return (pose['x'], pose['y'], pose['yaw'])
    return None


def arena_to_odom_broadcasters(ld: LaunchDescription):
    """Nodes in the graph that could publish an arena-frame -> `odom` edge.

    Recognises a `static_transform_publisher` naming the arena frame (the shape
    task 6.2/6.3 introduce), and any node parameterised with an arena
    `world_frame`/`global_frame`/`map_frame` (the shape a second
    `robot_localization` instance would take - design.md notes it could later
    replace the static broadcaster with no `frame_id` changes).
    """
    found = []
    for node in launch_nodes(ld):
        args = node_arguments(node)
        if node_executable(node) == 'static_transform_publisher':
            frames = {a.strip('-') for a in args}
            if ARENA_FRAME in frames or 'arena' in frames:
                found.append(node)
                continue
        for params in node_parameter_dicts(node):
            for key, value in params.items():
                if str(key).endswith(('world_frame', 'global_frame', 'map_frame')) \
                        and str(value) in (ARENA_FRAME, 'arena'):
                    found.append(node)
    return found


def published_arena_transform(ld: LaunchDescription):
    """`T_arena_odom` as the running system publishes it, or `None` if unpublished.

    Nothing can compose an arena pose without this edge, which is exactly
    requirement 1.3's defect.
    """
    for node in arena_to_odom_broadcasters(ld):
        args = node_arguments(node)
        if '--x' in args and '--y' in args and '--yaw' in args:
            return (float(args[args.index('--x') + 1]),
                    float(args[args.index('--y') + 1]),
                    float(args[args.index('--yaw') + 1]))
    return None


# --------------------------------------------------------------------------
# Config-file introspection
# --------------------------------------------------------------------------

def load_config(name: str):
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def controller_params(config_name: str, controller: str):
    """`ros__parameters` for one controller out of a ros2_control YAML."""
    cfg = load_config(config_name)
    return (cfg.get(controller) or {}).get('ros__parameters', {})


def spawned_controllers(ld: LaunchDescription):
    """Controller names a `spawner` node loads (its positional arguments)."""
    names = []
    for node in launch_nodes(ld):
        if node_executable(node) != 'spawner':
            continue
        for arg in node_arguments(node):
            if arg.startswith('-'):
                break
            names.append(arg)
    return names


# --------------------------------------------------------------------------
# Source (AST) introspection
# --------------------------------------------------------------------------

def parse_script(name: str) -> ast.Module:
    return ast.parse((SCRIPTS_DIR / name).read_text())


def frame_id_literals_by_method(source: Path):
    """`{method_name: [(lineno, frame_id_literal), ...]}` for every constant
    `*.frame_id = '...'` assignment in a Python source file."""
    tree = ast.parse(Path(source).read_text())
    out = {}
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == 'frame_id':
                    out.setdefault(func.name, []).append((node.lineno, node.value.value))
    return out
