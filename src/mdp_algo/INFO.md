# Codebase Overview

This is a Python algorithm package for an autonomous car/robot in a **40 × 40 grid arena**.

## Main Flow

```text
Obstacle input
  → OccupancyMap (obstacle/safety zones)
  → Hamiltonian (visit order)
  → Checkpoint selection (where to stop to scan each obstacle)
  → Hybrid A* (collision-aware car path)
  → pathcommands (compact robot commands + JSON)
  → Optional Pygame Simulator (visualisation)
```

The primary external libraries are `numpy`, `pygame`, and—optionally for occupancy-map debugging—`matplotlib`.

## Root Files

### `__init__.py`

Empty marker file that makes this directory importable as the `algo` Python package.

### `constants.py`

Central numeric configuration:

- A 40 × 40 grid representing a 200 cm × 200 cm arena.
- Pygame screen/map dimensions and offsets.
- Physical car properties, including the turning radius (26.75 cm) and rear-axle offset.

### `enumerations.py`

Defines named integer states:

- `CarState`: high-level states such as `START`, `DRIVE`, and image recognition.
- `Gear`: forward, park, and reverse.
- `Steering`: left, straight, and right.

### `utils.py`

Shared geometry and conversion helpers:

- Grid coordinate ↔ real-world coordinate conversion.
- Real-world coordinate → Pygame pixel conversion.
- Cardinal directions ↔ radians.
- Distance heuristics: Manhattan (`l1`), Euclidean (`l2`), and diagonal.
- Angle normalization and polar-coordinate helpers.
- Coordinate-frame transformation used by Reeds–Shepp calculations.

## `objects/`

Models for physical and map entities.

### `objects/Agent.py`

Defines `Car`, a high-level vehicle object holding pose, velocity, state, next checkpoint, and driving commands.

Most behaviour methods—such as driving, task control, and state updates—are stubs marked `pass`. This is planned abstraction code rather than the active planner.

### `objects/Border.py`

Pygame visual sprites:

- `Border`: visible black arena edge.
- `VirtualBorderWall`: transparent-red safety/collision overlay along the arena edge.

### `objects/Checkpoint.py`

Represents a destination pose for image recognition:

- Stores grid and continuous coordinates.
- Stores the required final orientation.
- Tracks whether the checkpoint has been completed.

### `objects/DriveCommand.py`

A simple data container for one driving instruction: gear, steering direction, and distance.

### `objects/Obstacle.py`

Pygame representation of arena obstacles:

- Loads and rotates `images/obstacle.png` according to the image-facing direction.
- Stores grid position, orientation, and ID.
- `VirtualWall` creates a larger transparent safety area around an obstacle, approximating required vehicle clearance.

### `objects/OccupancyMap.py`

The collision map used by planning:

- Maintains a 40 × 40 binary occupancy grid.
- Marks outer boundaries as blocked, except for the start-zone opening.
- Inflates each obstacle by roughly three cells in every direction for vehicle clearance.
- `collide_with_point(x, y)` converts a continuous point to grid space and returns whether it is occupied.

### `objects/images/obstacle.png`

Bitmap artwork used to display an obstacle in the simulator.

## `pathfinding/`

The route-planning logic.

### `pathfinding/hamiltonian.py`

Chooses the order in which to visit obstacles—effectively a travelling-salesperson-style layer.

- `find_brute_force_path()` checks every obstacle ordering and chooses the shortest; this is practical only for small obstacle counts.
- `find_nearest_neighbor_path()` uses a faster greedy nearest-obstacle strategy.
- `obstacle_to_checkpoint()` and `obstacle_to_checkpoint_all()` find valid camera/scanning positions in front of an obstacle while avoiding map collisions.
- Also includes image-orientation helpers and random/example map utilities.

### `pathfinding/hybrid_astar.py`

The primary local motion planner.

- `Node` represents one vehicle pose `(x, y, θ)`, its parent, cost, and preceding driving action.
- `HybridAStar.find_path()` expands six possible actions: forward/reverse × left/straight/right.
- Applies car-like arc motion using the minimum turning radius.
- Avoids occupied locations, discretizes poses for search bookkeeping, and rebuilds the route once it reaches a compatible final pose.
- Supports Euclidean, Manhattan, diagonal, and Reeds–Shepp heuristics.
- Penalizes gear and steering changes to encourage smoother paths.

### `pathfinding/reeds_shepp.py`

Implements Reeds–Shepp paths: shortest theoretical paths for a car that can drive both forward and reverse.

- `PathElement` stores one arc/straight segment, steering direction, and gear.
- Generates 12 standard Reeds–Shepp path families and reflected/time-reversed variants.
- Selects the shortest valid candidate.
- Used mainly as a path-length heuristic; it does not perform obstacle collision checks.

### `pathfinding/pathcommands.py`

Adapts a planned sequence of `HybridAStar` nodes into robot-facing output.

- Combines consecutive identical driving actions into compact commands such as `SFxxx` and `LBxxx`.
- Produces a simplified display path for an Android/client application.
- Wraps output in a navigation JSON payload:

  ```json
  {"type": "NAVIGATION", "data": {"commands": "...", "path": "..."}}
  ```

- `call_algo(message)` is an older one-call entry point that parses a `START_TASK`-style message, plans routes, and returns JSON.

### `pathfinding/task1.py`

Higher-level controller for the exploration/image-recognition task.

- Parses obstacle data, converts the coordinate/direction convention, and builds the occupancy map.
- Uses nearest-neighbour obstacle ordering.
- Tries multiple valid scanning checkpoints until Hybrid A* finds a route.
- Queues one command/path response per obstacle.
- Tracks obstacle IDs and image-recognition results.
- This is the clearest current application-level entry point for Task 1.

### `pathfinding/task2.py`

A partial, mostly conceptual controller for a second task: driving around two obstacles based on distance sensing and image-recognition direction.

- Contains calibrated constants and the intended control sequence.
- `measure_distance()` and `image_rec_direction()` are placeholders.
- Several manoeuvres remain `pass`, so this file is not operational yet.

## `simulation/`

Visualisation and sample scenarios.

### `simulation/simulator.py`

Pygame visual simulator.

- Draws the 40 × 40 grid map, 2.0 m × 2.0 m movement area, start zone, borders, obstacles, virtual safety walls, and planned routes.
- Shows a high-contrast yellow arrow on each obstacle's image-facing side.
- Animates a heading-aware robot along the planned path, including forward/reverse state and turns.
- Uses an exact shortest-time Hamiltonian visit order by default, with a five-image, timed recognition demonstration.
- Uses `Hamiltonian` to choose order and `HybridAStar` for each route segment.
- Colours forward and reverse path points differently and marks final scanning positions.
- Its `__main__` block runs one of the sample maps from `testing.py`.

### `simulation/testing.py`

Defines `get_maps()`, which returns ten hard-coded obstacle configurations for manual planner and simulator testing.

## Practical Status

The usable path-planning pipeline is primarily:

```text
task1.py → OccupancyMap → Hamiltonian → HybridAStar → pathcommands.py
```

The Pygame simulator provides visual validation and route animation. `Agent.py` and `task2.py` remain unfinished scaffolding.
