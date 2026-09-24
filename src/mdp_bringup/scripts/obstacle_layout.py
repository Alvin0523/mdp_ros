"""Task 1 obstacle layout - the one file both the simulator and the robot read.

A layout YAML (config/test_obstacles.yaml) lists obstacles the way the tablet
places them: a grid CELL (cell_x, cell_y: 0..19, 10 cm each, (0, 0) the
bottom-left cell) plus the facing side of the image. Metres (`x:`/`y:`) are
still accepted and snapped to the cell they fall in. From that one list:

  * `tablet_lines()`   - the exact OBSTACLE/DONE lines the tablet sends, for
                         fake_tablet.py to feed through bluetooth_bridge_node,
  * `setup_string()`   - the /obstacle_setup message that bridge produces, for
                         publish_test_obstacles.py (`pixi run setup`),
  * `world_sdf()`      - the Gazebo arena with the obstacles baked in.

The block fills its cell, so its centre is the cell corner + 5 cm - the same
conversion bluetooth_bridge_node applies to the tablet's corner coordinates.

Not an entry point: imported by the scripts above and by mdp.launch.py.
"""

from dataclasses import dataclass
import math
from typing import List, Optional

import yaml

CELL_CM = 10
GRID_CELLS = 20
FACINGS = ('N', 'E', 'S', 'W')


@dataclass(frozen=True)
class LayoutObstacle:
    id: int
    col: int
    row: int
    facing: str
    symbol: Optional[str] = None   # models/symbols/<symbol>.obj, sim only

    @property
    def centre_m(self):
        return ((self.col * CELL_CM + CELL_CM / 2.0) / 100.0,
                (self.row * CELL_CM + CELL_CM / 2.0) / 100.0)


def load(path: str) -> List[LayoutObstacle]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out = []
    for entry in data.get('obstacles') or []:
        if 'cell_x' in entry and 'cell_y' in entry:
            col, row = int(entry['cell_x']), int(entry['cell_y'])
        else:
            col = int(math.floor(float(entry['x']) * 100.0 / CELL_CM + 1e-6))
            row = int(math.floor(float(entry['y']) * 100.0 / CELL_CM + 1e-6))
        ob = LayoutObstacle(
            id=int(entry['id']), col=col, row=row,
            facing=str(entry['facing']).strip().upper(), symbol=entry.get('symbol'))
        if not (0 <= ob.col < GRID_CELLS and 0 <= ob.row < GRID_CELLS):
            raise ValueError(f'{path}: obstacle {ob.id} cell ({ob.col}, {ob.row}) is off the 20x20 grid')
        if ob.facing not in FACINGS:
            raise ValueError(f'{path}: obstacle {ob.id} facing {ob.facing!r} is not N/E/S/W')
        out.append(ob)
    return out


def tablet_lines(obstacles: List[LayoutObstacle]) -> List[str]:
    """What the tablet sends: one OBSTACLE line per block (cell corner in cm), then DONE."""
    lines = [f'OBSTACLE,{o.id},{o.col * CELL_CM},{o.row * CELL_CM},{o.facing}' for o in obstacles]
    return lines + ['DONE']


def describe(obstacles: List[LayoutObstacle]) -> str:
    """Log line in grid cells: '#1 (5,10) S | #2 (12,4) W'."""
    return ' | '.join(f'#{o.id} ({o.col},{o.row}) {o.facing}' for o in obstacles)


def setup_string(obstacles: List[LayoutObstacle]) -> str:
    """What bluetooth_bridge_node publishes on /obstacle_setup for the same set."""
    return '|'.join(f'{o.id}:{o.centre_m[0]:.2f},{o.centre_m[1]:.2f},{o.facing}' for o in obstacles)


# ---------------------------------------------------------------- Gazebo ----
#
# Geometry, for anyone changing the model below:
#   Obstacle - 10 x 10 cm footprint, 15 cm tall. Model z = 0.075 (height/2) so it
#   rests on the floor and the link origin is the cube's centre.
#   Symbol decal - models/symbols/<symbol>.obj is already the real 6.1 cm print,
#   so scale 1. Face offset 0.0515 = 0.05 (half width) + 0.001 (half the 2 mm
#   panel) + 0.0005 clearance; z 0.0445 = 0.075 - 0.061/2 puts it flush with
#   the top. Rotation is YAW ONLY - the quad is modelled upright facing North,
#   so yaw swings it to any face without disturbing image-up.
#
# Obstacles must exist at world load: spawned afterwards, the decal texture
# never binds in ogre2 and renders black. Hence baking them into the SDF text.

WORLD_PLACEHOLDER = '<!-- OBSTACLES -->'

_DECAL = {  # facing -> (dx, dy, yaw)
    'N': (0.0, 0.0515, 0.0),
    'S': (0.0, -0.0515, math.pi),
    'E': (0.0515, 0.0, -math.pi / 2.0),
    'W': (-0.0515, 0.0, math.pi / 2.0),
}


def _model_sdf(o: LayoutObstacle) -> str:
    x, y = o.centre_m
    decal = ''
    if o.symbol:
        dx, dy, yaw = _DECAL[o.facing]
        decal = f'''
        <visual name="symbol">
          <pose>{dx} {dy} 0.0445 0 0 {yaw}</pose>
          <geometry>
            <mesh>
              <uri>model://mdp_description/models/symbols/{o.symbol}.obj</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </visual>'''
    return f'''    <model name="obstacle_{o.id}">
      <static>true</static>
      <pose>{x} {y} 0.075 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.1 0.1 0.15</size></box></geometry>
        </collision>
        <visual name="body">
          <geometry><box><size>0.1 0.1 0.15</size></box></geometry>
          <material>
            <ambient>0.15 0.15 0.15 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>{decal}
      </link>
    </model>'''


def world_sdf(base_sdf: str, obstacles: List[LayoutObstacle]) -> str:
    """`base_sdf` with one model per obstacle inserted at WORLD_PLACEHOLDER."""
    if WORLD_PLACEHOLDER not in base_sdf:
        raise ValueError(f'arena SDF has no {WORLD_PLACEHOLDER} marker to insert obstacles at')
    return base_sdf.replace(WORLD_PLACEHOLDER, '\n'.join(_model_sdf(o) for o in obstacles))
