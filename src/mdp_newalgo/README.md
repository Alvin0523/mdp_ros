# Autonomous Robot Path Planning

This package plans collision-aware routes for a car-like robot in a 40 × 40 grid arena. It is designed for an exploration task where the robot must visit obstacles, stop at valid image-recognition positions, and produce compact driving commands.

The planner supports forward and reverse driving, left/right turns, obstacle safety margins, route ordering, and an optional Pygame visual simulator.

## How it works

```text
Obstacle message
    ↓
OccupancyMap: mark walls and obstacle safety zones
    ↓
Hamiltonian: choose an order for visiting obstacles
    ↓
Checkpoint search: choose a valid pose to scan each obstacle
    ↓
Hybrid A*: find a collision-free car path for every checkpoint
    ↓
pathcommands: combine movements and return navigation JSON
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `constants.py` | Arena, display, and vehicle measurements. |
| `enumerations.py` | Named gear, steering, and car-state values. |
| `utils.py` | Coordinate conversion, angle, and distance utilities. |
| `objects/` | Arena entities: obstacles, occupancy map, borders, checkpoints, and command models. |
| `pathfinding/` | Obstacle ordering, Hybrid A* planning, Reeds–Shepp heuristic, Task 1 controller, and command conversion. |
| `simulation/` | Pygame route visualizer and example obstacle maps. |

## Requirements

- Python 3.9 or newer
- `numpy`
- `pygame` for simulation and obstacle sprites
- `matplotlib` only for the optional occupancy-map debug display

Install the dependencies:

```bash
python -m pip install numpy pygame matplotlib
```

## Running the simulator

Run this command from the directory **containing** the `algo` package (normally the parent of this repository folder):

```bash
python -m algo.simulation.simulator
```

The simulator loads a sample obstacle map, computes an obstacle order and a route to each checkpoint, then displays the route in a Pygame window.

It renders the 40 × 40 grid, 2.0 m × 2.0 m arena, start zone, obstacles, image-facing arrows, safety zones, planned route, and an animated, heading-aware robot. Green and red route points indicate forward and reverse movements respectively.

The default simulator scenario contains five images. It computes an exact shortest-time Hamiltonian order from collision-aware Hybrid A* route legs. The cost model includes forward/reverse speeds, steering and gear-change delays, and one second of recognition time per image. It starts a 120-second simulated timer, pauses for recognition at each reached checkpoint, marks recognised images with a green tick, and displays the live recognition count, predicted route time, and final result. Visual playback defaults to a demo-friendly 1.5× real time.

The sample selected in `simulation/simulator.py` can be changed by editing the index used with `get_maps()`.

## Using Task 1 programmatically

`pathfinding/task1.py` contains the application-level controller for exploration. It accepts a `START_TASK`-style dictionary, plans all reachable obstacle visits, and makes one navigation response available at a time.

```python
from algo.pathfinding.task1 import task1

message = {
    "type": "START_TASK",
    "data": {
        "task": "EXPLORATION",
        "robot": {"id": "R", "x": 1, "y": 1, "dir": "N"},
        "obstacles": [
            {"id": "00", "x": 8, "y": 5, "dir": "S"},
            {"id": "01", "x": 10, "y": 17, "dir": "W"},
        ],
    },
}

planner = task1()
planner.generate_path(message)

while not planner.has_task_ended():
    navigation = planner.get_command_to_next_obstacle()
    obstacle_id = planner.get_obstacle_id()
    # Send `navigation` to the robot/client, then process obstacle_id.
```

Incoming 20 × 20-grid coordinates are converted to centimetres internally. Obstacle directions are interpreted directly as the image-bearing side: `E`, `N`, `W`, or `S`.

## Navigation response

Each planned leg returns a dictionary in this form:

```json
{
  "type": "NAVIGATION",
  "data": {
    "commands": ["SF020", "LF045", "SB010"],
    "path": [[1, 1], [1, 2], [2, 2]]
  }
}
```

Commands use the following compact format:

| Prefix | Meaning |
| --- | --- |
| `SF` / `SB` | Drive straight forward / backward; value is distance. |
| `LF` / `LB` | Turn left while moving forward / backward; value is angle. |
| `RF` / `RB` | Turn right while moving forward / backward; value is angle. |

The exact command grouping is produced by `construct_path_2()` in `pathfinding/pathcommands.py`.

## Core planning details

- The arena is 200 cm × 200 cm and is represented by a 40 × 40 grid.
- Obstacles and map borders are expanded into virtual walls to provide vehicle clearance.
- `Hamiltonian` supports exhaustive and nearest-neighbour obstacle ordering; Task 1 uses nearest-neighbour ordering.
- `HybridAStar` expands six primitive motions: forward/reverse × left/straight/right.
- The planner can penalize gear and steering changes to prefer smoother paths.
- Reeds–Shepp path lengths are available as an obstacle-unaware heuristic.

## Current limitations

- `objects/Agent.py` is a scaffold: its vehicle state-machine methods are not implemented.
- `pathfinding/task2.py` is a partial prototype and contains placeholder sensor and manoeuvre logic.
- There is currently no automated test suite or dependency lockfile.
- Several modules rely on package-style imports (`algo.*`), so module execution from the package parent is the most reliable way to run them.

## Useful files to start with

- For route planning: `pathfinding/task1.py`
- For the A* implementation: `pathfinding/hybrid_astar.py`
- For obstacle ordering/checkpoints: `pathfinding/hamiltonian.py`
- For robot command output: `pathfinding/pathcommands.py`
- For visual debugging: `simulation/simulator.py`
