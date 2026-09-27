"""Obstacle layouts - config/tasks.yaml, the one file the simulator and the robot read.

  task1  obstacles by tablet CELL (cell_x, cell_y: 0..19, 10 cm each, (0, 0)
         the bottom-left cell) plus the image side. Metres (`x:`/`y:`) are
         still accepted and snapped to the cell they fall in.
  task2  obstacles in metres with their size.

From one list:
  * `setup_string()`  - the /obstacle_setup message bluetooth_bridge_node
                        produces for the same set (task 1: `pixi run setup`,
                        obstacles:=yaml at launch),
  * `world_sdf()`     - the Gazebo arena with the obstacles baked in,
  * `model_sdf()`     - one obstacle, for spawning at runtime (sim_obstacles),
  * `from_setup_string()` - the other way: a tablet set back to obstacles.

A task 1 block fills its cell, so its centre is the cell corner + 5 cm - the
same conversion bluetooth_bridge_node applies to the tablet's corner coordinates.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import yaml

CELL_M = 0.10
GRID_CELLS = 20
FACINGS = ('N', 'E', 'S', 'W')
TASK1_SIZE = (0.10, 0.10, 0.15)    # the task 1 block: 10 x 10 cm, 15 cm tall
SYMBOL_M = 0.061                   # printed symbol, square


@dataclass(frozen=True)
class LayoutObstacle:
    id: int
    x: float                       # centre, metres, arena frame
    y: float
    facing: str                    # N/E/S/W: the side with the image
    size: Tuple[float, float, float] = TASK1_SIZE
    symbol: Optional[str] = None   # models/symbols/<symbol>.obj, sim only

    @property
    def cell(self) -> Tuple[int, int]:
        return (int(math.floor(self.x / CELL_M + 1e-6)), int(math.floor(self.y / CELL_M + 1e-6)))


def _cell_centre(c: int) -> float:
    return round((c + 0.5) * CELL_M, 4)


def load(path: str, task: str = 'task1') -> List[LayoutObstacle]:
    with open(path) as f:
        entries = ((yaml.safe_load(f) or {}).get(task) or {}).get('obstacles') or []
    out = []
    for e in entries:
        facing = str(e['facing']).strip().upper()
        if facing not in FACINGS:
            raise ValueError(f"{path}: {task} obstacle {e['id']} facing {facing!r} is not N/E/S/W")
        if task == 'task1':
            if 'cell_x' in e:
                col, row = int(e['cell_x']), int(e['cell_y'])
            else:
                col, row = (int(math.floor(float(e[k]) / CELL_M + 1e-6)) for k in ('x', 'y'))
            if not (0 <= col < GRID_CELLS and 0 <= row < GRID_CELLS):
                raise ValueError(f"{path}: task1 obstacle {e['id']} cell ({col}, {row}) is off the 20x20 grid")
            ob = LayoutObstacle(int(e['id']), _cell_centre(col), _cell_centre(row), facing,
                                symbol=e.get('symbol'))
        else:
            ob = LayoutObstacle(int(e['id']), float(e['x']), float(e['y']), facing,
                                tuple(float(v) for v in e['size']), e.get('symbol'))
        out.append(ob)
    return out


def describe(obstacles: List[LayoutObstacle]) -> str:
    """Log line in grid cells: '#1 (5,10) S | #2 (12,4) W'."""
    return ' | '.join(f'#{o.id} ({o.cell[0]},{o.cell[1]}) {o.facing}' for o in obstacles)


def setup_string(obstacles: List[LayoutObstacle]) -> str:
    """What bluetooth_bridge_node publishes on /obstacle_setup for the same set."""
    return '|'.join(f'{o.id}:{o.x:.2f},{o.y:.2f},{o.facing}' for o in obstacles)


def from_setup_string(data: str) -> List[LayoutObstacle]:
    """An /obstacle_setup message ('1:0.55,1.05,S|...') as task 1 blocks, no symbols.
    Entries task1_runner would skip are skipped here too."""
    out = []
    for item in data.strip().split('|'):
        try:
            oid, rest = item.split(':')
            x, y, facing = rest.split(',')[:3]
            ob = LayoutObstacle(int(oid), float(x), float(y), facing.strip().upper())
        except ValueError:
            continue
        if ob.facing in FACINGS:
            out.append(ob)
    return out


# ---------------------------------------------------------------- Gazebo ----
#
# Geometry, for anyone changing the model below: the model sits at z = height/2
# so it rests on the floor and the link origin is the box's centre. The symbol
# decal - models/symbols/<symbol>.obj, already the real 6.1 cm print, scale 1 -
# is 1.5 mm off the face (half the 2 mm panel + 0.5 mm clearance) and flush with
# the top. Rotation is YAW ONLY: the quad is modelled upright facing North, so
# yaw swings it to any face without disturbing image-up.
#
# Obstacles that exist at world load get their decal textures; spawned
# afterwards the decal texture never binds in ogre2 and renders black. Hence
# baking the layout into the SDF text, and runtime spawning only for a tablet
# layout that differs from the file.

WORLD_PLACEHOLDER = '<!-- OBSTACLES -->'
_DECAL_YAW = {'N': 0.0, 'S': math.pi, 'E': -math.pi / 2.0, 'W': math.pi / 2.0}
_FACING_DIR = {'N': (0.0, 1.0), 'S': (0.0, -1.0), 'E': (1.0, 0.0), 'W': (-1.0, 0.0)}


def model_sdf(o: LayoutObstacle, sdf_root: bool = False) -> str:
    """One obstacle as an SDF <model> (sdf_root: wrapped in <sdf>, for spawning)."""
    sx, sy, sz = o.size
    decal = ''
    if o.symbol:
        fx, fy = _FACING_DIR[o.facing]
        dx, dy = fx * (sx / 2.0 + 0.0015), fy * (sy / 2.0 + 0.0015)
        decal = f'''
        <visual name="symbol">
          <pose>{dx:.4f} {dy:.4f} {sz / 2.0 - SYMBOL_M / 2.0:.4f} 0 0 {_DECAL_YAW[o.facing]}</pose>
          <geometry>
            <mesh>
              <uri>model://mdp_description/models/symbols/{o.symbol}.obj</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </visual>'''
    model = f'''    <model name="obstacle_{o.id}">
      <static>true</static>
      <pose>{o.x} {o.y} {sz / 2.0} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        </collision>
        <visual name="body">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material>
            <ambient>0.15 0.15 0.15 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>{decal}
      </link>
    </model>'''
    return f'<sdf version="1.8">\n{model}\n</sdf>' if sdf_root else model


def world_sdf(base_sdf: str, obstacles: List[LayoutObstacle]) -> str:
    """`base_sdf` with one model per obstacle inserted at WORLD_PLACEHOLDER."""
    if WORLD_PLACEHOLDER not in base_sdf:
        raise ValueError(f'arena SDF has no {WORLD_PLACEHOLDER} marker to insert obstacles at')
    return base_sdf.replace(WORLD_PLACEHOLDER, '\n'.join(model_sdf(o) for o in obstacles))
