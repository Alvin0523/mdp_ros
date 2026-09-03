# Notes to Take: Simulator and Path-Planning Changes

This file is documentation only. It does not change the algorithm or simulator behaviour.

## Important: Shortest-Time Assumptions

Those values were my modelling assumptions—not values supplied by your assignment, robot specification, or original code. I should have made that explicit.

The original project does not define real robot speeds or manoeuvre durations. It only has a placeholder `dt = 0.1` in `Agent.py`, with movement methods unimplemented. I introduced these values to create a usable time-cost model:

- `20 cm/s` forward: assumed nominal drive speed.
- `15 cm/s` reverse: assumed slower than forward for safety.
- `0.5 s` gear change: assumed pause for switching direction.
- `0.15 s` steering change: assumed steering-actuation overhead.
- `1.0 s` recognition: assumed camera/image-processing dwell time.

Therefore, the reported `34.84 seconds` is only valid under those assumptions. It is not an official “shortest time” for your physical robot.

To make this academically correct, we should replace them with your project’s calibrated/measured values, such as motor-command speed, forward/reverse turn timings, gear-change delay, and image-recognition time.

## Original Codebase Findings

Before the changes, this project contained:

- A 40 × 40 occupancy grid that represents a 200 cm × 200 cm arena.
- A nearest-neighbour and brute-force obstacle ordering implementation.
- A Hybrid A* car-like route planner.
- A Pygame path visualiser that could draw obstacles, safety overlays, and completed paths.

It did **not** contain:

- A visible coordinate grid or axis labels.
- A robot visualisation or movement animation.
- A visual indicator for the image-facing side of an obstacle.
- Image-recognition events or recognition progress.
- A task timer or timeout result.
- A five-image default test scenario.
- A shortest-time model based on speeds and manoeuvre delays.
- Consistent package imports for the documented module launch command.

## Complete Change Record

### New documentation and setup files

#### `README.md` — added

Added a project README that documents:

- Project purpose and architecture.
- Repository/folder layout.
- Required packages.
- Virtual-environment setup and simulator launch command.
- Task 1 input/output use.
- Robot command formats.
- Simulator functionality and known project limitations.
- The timed five-image shortest-time simulation model and its assumptions.

#### `INFO.md` — added

Added a formatted codebase overview that documents every existing root file, object model, pathfinding module, simulation module, and the practical execution flow.

#### `requirements.txt` — added

Added the dependency list:

```text
numpy>=1.24
pygame>=2.5
matplotlib>=3.7
```

#### `.venv/` — created locally

Created a Python virtual environment and installed the packages in `requirements.txt`.

The environment directory is intentionally not included in Git status because it is a local machine environment, not project source code.

### Package-import compatibility fixes

The code mixed package imports such as `algo.objects...` with local imports such as `objects...`. This prevented the documented command below from launching correctly:

```bash
python -m algo.simulation.simulator
```

The following files were changed to consistently use package-qualified imports:

| File | Import changes |
| --- | --- |
| `simulation/simulator.py` | Uses `algo.enumerations`, `algo.objects`, `algo.pathfinding`, `algo.simulation`, `algo.utils`, and `algo.constants`. |
| `objects/OccupancyMap.py` | Uses `algo.objects.Obstacle`, `algo.utils`, and `algo.simulation.testing`. |
| `pathfinding/hamiltonian.py` | Uses `algo.utils`, `algo.objects.Obstacle`, `algo.enumerations`, `algo.pathfinding.reeds_shepp`, and `algo.constants`. |
| `pathfinding/hybrid_astar.py` | Uses `algo.enumerations`, `algo.objects`, `algo.utils`, `algo.constants`, and `algo.pathfinding.reeds_shepp`. |
| `pathfinding/reeds_shepp.py` | Uses `algo.enumerations`, `algo.utils`, and `algo.constants`. |
| `simulation/testing.py` | Uses `algo.objects.Obstacle`. |

### `simulation/simulator.py` changes

#### Pygame initialisation

- Moved `pygame.init()` to occur before display and font creation.
- Added `font` and `small_font` resources for readable UI text.

#### 40 × 40 movement-area grid

- Added `draw_grid()`.
- Draws all 40 vertical and 40 horizontal cells.
- Uses the existing 200 cm × 200 cm arena dimensions.
- Added `draw_axes()`.
- Displays X and Y grid labels from `0` to `40`, in increments of `5`.
- Labels the axes as grid-cell coordinates.

#### Cleaner visual layout

- Clears the full window every frame to prevent stale robot graphics/text.
- Replaced scattered coloured labels with a white, bordered information panel.
- Added a legend for image-facing arrows, safety areas, and robot front direction.
- Added movement-area title and dimensions: `2.0 m × 2.0 m | 40 × 40 grid`.
- Retained the visible start zone, arena borders, obstacles, and red safety/no-go overlays.

#### Image-facing markers

- Added `draw_image_marker(obstacle)`.
- Draws a yellow arrow with a black outline in the direction of the obstacle’s image-facing side.
- This makes image positions/orientations visible independently from the obstacle sprite rotation.

#### Robot rendering and animation

- Added `draw_robot(x, y, theta, action)`.
- Draws a blue car-shaped robot at the calculated rear-axle pose.
- Draws a yellow pointed nose, making the robot front and heading unambiguous.
- Draws the robot path frame-by-frame rather than only showing a final route.
- Shows current grid pose, heading, gear, and steering state in the status panel.
- Forward and reverse path nodes remain colour-coded green and red.

#### Recognition-state visualisation

- Added `draw_obstacle_state(obstacle)`.
- Pending images show only their obstacle and image-facing arrow.
- The image currently being recognised shows an orange circular marker.
- Recognised images show a green marker with a white tick.
- Unreachable images show a red marker with a white cross.

#### Timed task state

- Added task fields for timeout, recognition duration, target image count, recognition set, unreachable-image set, task state, and predicted route time.
- Added `complete_recognition(obstacle)`.
- Each obstacle is stored by object identity in a set, so it can only be recognised once.
- Added `update_task(delta_seconds)`.
- Implements the sequence:

  ```text
  move to next path node
    → reach image checkpoint
    → pause for recognition duration
    → mark image recognised
    → move to next image
  ```

- Implements the task outcomes:
  - `PLANNING`
  - `MOVING`
  - `RECOGNISING`
  - `COMPLETE`
  - `TIME EXPIRED`
  - `INCOMPLETE`

- Added `draw_task_status()` with:
  - time remaining;
  - recognised-image count;
  - ordering method;
  - predicted route time;
  - task status;
  - robot position/heading/motion; and
  - unreachable-image count.

#### Default simulator scenario

- Changed the simulator’s default map from eight obstacles to five obstacles:

  ```python
  map = test_maps[7][:5]
  ```

- Configured the default run with:

  ```python
  time_limit_seconds=120
  recognition_seconds=1
  target_images=5
  ordering_method='shortest_time'
  ```

### `pathfinding/hamiltonian.py` changes

#### Robust obstacle reachability

- Added `self.unreachable_obstacles`.
- Added `_reachable_obstacles()`.
- It checks whether each obstacle has a valid collision-free scanning checkpoint before route ordering.
- The prior nearest-neighbour logic removed items during iteration; it now uses a safe reachable list and reports obstacles without valid checkpoints.

#### Exact distance-based ordering retained

- `find_brute_force_path()` now handles zero reachable obstacles safely and returns a list consistently.
- It remains available as an exact **distance-based** comparison mode.

#### Physical travel-time calculation

Added:

```python
path_travel_time(path, L, forward_speed, reverse_speed,
                 gear_change_time, steering_change_time)
```

For every Hybrid A* action/node, it adds:

```text
motion duration = command distance / gear-specific speed
plus gear-change delay when gear changes
plus steering-change delay when steering changes
```

#### Exact shortest-time Hamiltonian computation

Added:

```python
find_shortest_time_hamiltonian(...)
```

It performs the following process for the five-image task:

```text
1. Find a valid recognition checkpoint for each image.
2. Use Hybrid A* to plan every directed leg:
   start → image
   image → image
3. Convert every collision-aware leg into calibrated seconds.
4. Test every image order (5! = 120 permutations).
5. Add one recognition dwell time for every image.
6. Choose the route with the smallest total modelled time.
7. Return the selected obstacle order, Hybrid A* paths,
   predicted total seconds, and unreachable images.
```

This is the function used by the default simulator mode.

### `pathfinding/hybrid_astar.py` changes

#### Physical-time cost model

Extended `HybridAStar.__init__()` with these optional arguments:

```python
cost_mode='distance'
forward_speed=20.0
reverse_speed=15.0
gear_change_time=0.5
steering_change_time=0.15
```

Added:

```python
transition_cost(previous_action, action)
```

In `cost_mode='time'`, this uses seconds instead of distance as the search cost.

#### A* correctness improvements

- Changed the goal test so a goal is accepted when it is the best queued node being expanded, rather than when it is first generated as a child.
- Accumulates transition cost in `childNode.g`.
- Uses a straight-line maximum-speed lower bound as a time heuristic.
- Compares open/closed records using accumulated `g` cost instead of `f` cost.
- Retains distance-based mode for the pre-existing paths that do not request time optimisation.

### Calibrated animation timing

The simulator no longer uses a constant `0.12 s` per path node as the task clock.

It now builds `node_durations` from the same physical-time model used during route optimisation. Therefore:

- forward nodes animate according to forward speed;
- reverse nodes animate according to reverse speed;
- gear/steering changes add their configured delays; and
- recognition pauses add recognition duration.

`playback_speed=5.0` is used to display the simulated route five times faster than real time. The countdown still advances in simulated physical seconds.

## Validation Performed

The following checks were run after the changes:

```text
Python compilation: passed
Git whitespace check: passed
Headless Pygame simulator construction: passed
Headless rendering-helper check: passed
Recognition state-machine check: passed
```

Shortest-time planner check on the default five-image scenario:

```text
5 unique image visits
5 collision-aware planned legs
0 unreachable images
Predicted shortest modelled route time: 34.84 seconds
```

## How to Run

```bash
cd /Users/rashiojha/SC2079-MDP-Group-29/algo
source .venv/bin/activate
cd ..
python -m algo.simulation.simulator
```

## Values That Must Be Calibrated Before Final Submission

Replace these assumptions with measured values from the actual robot:

| Setting | Current assumed value | Needed source |
| --- | ---: | --- |
| Forward speed | 20 cm/s | Measured straight-forward command speed. |
| Reverse speed | 15 cm/s | Measured straight-reverse command speed. |
| Gear-change time | 0.5 s | Measured direction-switch delay. |
| Steering-change time | 0.15 s | Measured steering actuation delay. |
| Recognition time | 1.0 s | Measured camera/image-recognition duration. |
| Timeout | 120 s | The official task time limit. |

Until calibration is supplied, describe the result as the **shortest route under the stated simulation assumptions**, rather than the verified shortest real-world robot time.
