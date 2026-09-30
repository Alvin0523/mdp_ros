"""Obstacle layouts - config/tasks.yaml, the one file the simulator and the robot read.

  task1  obstacles by tablet CELL (cell_x, cell_y: 0..19, 10 cm each, (0, 0)
         the bottom-left cell) plus the image side. Metres (`x:`/`y:`) are
         still accepted and snapped to the cell they fall in.
  task2  the course rules (carpark, obstacle sizes) - `load_task2()` - and a
         sim-only layout (distances, arrows) - `task2_sim()`.

From one list:
  * `setup_string()`  - the /obstacle_setup message bluetooth_bridge_node
                        produces for the same set (task 1: `pixi run setup`,
                        obstacles:=yaml at launch),
  * `world_sdf()`     - the Gazebo arena with the obstacles baked in,
  * `model_sdf()`     - one obstacle, for spawning at runtime (sim_helpers),
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
    """Task 1's obstacles (task 2 has load_task2 / task2_sim)."""
    assert task == 'task1', task
    with open(path) as f:
        entries = ((yaml.safe_load(f) or {}).get('task1') or {}).get('obstacles') or []
    out = []
    for e in entries:
        facing = str(e['facing']).strip().upper()
        if facing not in FACINGS:
            raise ValueError(f"{path}: task1 obstacle {e['id']} facing {facing!r} is not N/E/S/W")
        if 'cell_x' in e:
            col, row = int(e['cell_x']), int(e['cell_y'])
        else:
            col, row = (int(math.floor(float(e[k]) / CELL_M + 1e-6)) for k in ('x', 'y'))
        if not (0 <= col < GRID_CELLS and 0 <= row < GRID_CELLS):
            raise ValueError(f"{path}: task1 obstacle {e['id']} cell ({col}, {row}) is off the 20x20 grid")
        out.append(LayoutObstacle(int(e['id']), _cell_centre(col), _cell_centre(row), facing,
                                  symbol=e.get('symbol')))
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


# ------------------------------------------------------------------ task 2 ----
#
# Frame: x along the course (carpark -> obstacles), y across, centre line (the
# carpark's centre, both obstacles) at y = TASK2_CENTRE_Y. Chosen so the whole
# course has positive coordinates, like task 1's table - the planner's costmap
# starts at (0, 0).

TASK2_AREA_M = (5.00, 2.40)      # carpark to past obstacle 2 at the longest (d1 = d2 = 1.5 m), side walls included
TASK2_CENTRE_Y = 1.20
TASK2_BACK_WALL_X = 0.10         # inside face of the carpark's back wall
TASK2_SIDE_WALL_LENGTH = 1.20    # sim side walls, centred on obstacle 2
TASK2_SIDE_WALL_HEIGHT = 0.30
ARROW_SYMBOL = {'LEFT': '39_ArrowLeft', 'RIGHT': '38_ArrowRight'}


@dataclass(frozen=True)
class Task2Arena:
    """The task 2 course rules (config/tasks.yaml task2), metres."""
    carpark_width: float
    carpark_depth: float
    wall: float
    wall_height: float
    obstacle_1_size: Tuple[float, float, float]
    obstacle_2_size: Tuple[float, float, float]
    side_wall_clearance: float

    @property
    def opening_x(self) -> float:
        return TASK2_BACK_WALL_X + self.carpark_depth

    @property
    def carpark_centre(self) -> Tuple[float, float]:
        return TASK2_BACK_WALL_X + self.carpark_depth / 2.0, TASK2_CENTRE_Y

    def car_pose_centred(self, car_centre_ahead: float, heading: float):
        """base_link (rear axle) pose that puts the car's middle on the
        carpark's centre, facing `heading` (0 = out, pi = in).
        car_centre_ahead: how far the car's middle is ahead of base_link."""
        cx, cy = self.carpark_centre
        return cx - math.cos(heading) * car_centre_ahead, cy, heading

    def carpark_walls(self) -> List[Tuple[float, float, float, float]]:
        """The U of walls, (x0, y0, x1, y1): back wall + two sides, open toward +x."""
        b, w, t, cy = TASK2_BACK_WALL_X, self.carpark_width / 2.0, self.wall, TASK2_CENTRE_Y
        return [(b - t, cy - w - t, b, cy + w + t),                       # back
                (b, cy + w, self.opening_x, cy + w + t),                  # left side
                (b, cy - w - t, self.opening_x, cy - w)]                  # right side

    def side_walls(self, obstacle_2_x: float) -> List[Tuple[float, float, float, float]]:
        """The walls that MAY stand side_wall_clearance out from obstacle 2's ends."""
        y = self.obstacle_2_size[1] / 2.0 + self.side_wall_clearance
        x0, x1 = obstacle_2_x - TASK2_SIDE_WALL_LENGTH / 2.0, obstacle_2_x + TASK2_SIDE_WALL_LENGTH / 2.0
        cy, t = TASK2_CENTRE_Y, self.wall
        return [(x0, cy + y, x1, cy + y + t), (x0, cy - y - t, x1, cy - y)]


def load_task2(path: str) -> Task2Arena:
    with open(path) as f:
        t = (yaml.safe_load(f) or {})['task2']
    c = t['carpark']
    return Task2Arena(float(c['width']), float(c['depth']), float(c['wall']), float(c['height']),
                      tuple(float(v) for v in t['obstacle_1']['size']),
                      tuple(float(v) for v in t['obstacle_2']['size']),
                      float(t['side_wall_clearance']))


def task2_sim(path: str):
    """The sim-only layout: (obstacles with their arrows, walls) as LayoutObstacles
    for world_sdf(), and (obstacle_1_x, obstacle_2_x) centres."""
    arena = load_task2(path)
    with open(path) as f:
        sim = (yaml.safe_load(f) or {})['task2']['sim']
    s1, s2 = arena.obstacle_1_size, arena.obstacle_2_size
    x1 = arena.opening_x + float(sim['d1']) + s1[0] / 2.0
    x2 = x1 + s1[0] / 2.0 + float(sim['d2']) + s2[0] / 2.0
    cy = TASK2_CENTRE_Y
    obstacles = [
        LayoutObstacle(1, x1, cy, 'W', s1, ARROW_SYMBOL[str(sim['arrow_1']).upper()]),
        LayoutObstacle(2, x2, cy, 'W', s2, ARROW_SYMBOL[str(sim['arrow_2']).upper()]),
    ]
    rects = [(r, arena.wall_height) for r in arena.carpark_walls()]
    if sim.get('side_walls'):
        rects += [(r, TASK2_SIDE_WALL_HEIGHT) for r in arena.side_walls(x2)]
    walls = [LayoutObstacle(100 + i, (x0 + x1_) / 2.0, (y0 + y1) / 2.0, 'W', (x1_ - x0, y1 - y0, h))
             for i, ((x0, y0, x1_, y1), h) in enumerate(rects)]
    return obstacles, walls, (x1, x2)


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
