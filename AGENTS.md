# AGENTS.md — operating `mdp_ros`

ROS 2 Jazzy workspace for the MDP Ackermann car (NTU MDP, Group 14). The same code runs the real
car (Raspberry Pi 4 + STM32) and a Gazebo sim. Human-facing docs: `../docs/quickstart.md` (commands,
calibration, every measured number) and `../docs/rpi/`.

## Environment

- Everything runs through **pixi**. Never `pip install` or `source /opt/ros`: use `pixi run <task>` or
  `pixi run -- <command>`, from this folder.
- Build: `pixi run build` (`colcon build --symlink-install`). Python edits under `src/` need no
  rebuild; new files, new entry points, `setup.py` or `package.xml` changes do.
- After a package changes type or files are renamed: `pixi run clean && pixi run build`.
- Tests: `pixi run test`, or quicker:
  `pixi run -- python -m pytest -q src/mdp_bringup/test src/mdp_algorithm/test`.

## Layout

| Package | What |
| --- | --- |
| `mdp_bringup` (Python) | `launch/mdp.launch.py` (everything, sim and real), `launch/vision.launch.py`, `config/` (all settings). Nodes: `tasks/` (the two runners, sharing `runner_base.py`), `robot/` (pose feedback, manual drive, health, bag recorder, bt monitor), `sim/sim_helpers.py` (sim-only stand-ins, one node). CLI tools in `tools/` (`calib/` = straight, rotate, turn, goto, ultrasonic; rows go to `calibration_log.csv`) |
| `mdp_algorithm` (Python library, no nodes) | costmap (any area size), Hybrid A*, visit order, pure pursuit; `utils/params.py` loads settings |
| `mdp_vision` (Python) | `rpi_cam_publisher`, `yolo_detector`, models in `models/` |
| `mdp_bridge` (C++) | `serial_bridge_node` (STM32), `bluetooth_bridge_node` (tablet) |
| `mdp_description` (CMake) | `urdf/mdp_robot.urdf.xacro` (one URDF, `sim:=true/false`), arenas, symbol meshes |
| `mdp_interfaces` (CMake) | `RunStatus.msg` |

## Where each number lives: exactly one place

| Setting | File |
| --- | --- |
| Car size, wheel radius, steering limits, sensor positions | URDF xacro properties (plain numbers; `params.car_from_urdf()` parses them, the launch copies them into the controller and planner) |
| Footprint, turning circles, costmap, planner, follower, all speeds | `config/navigation.yaml` |
| Obstacle layouts (task 1 cells, task 2 metres) | `config/tasks.yaml` (Gazebo and the runners both read it) |
| ros2_control · EKF · devices · camera/YOLO | `config/controller.yaml` · `ekf.yaml` · `bridges.yaml` · `vision.yaml` |

Do not add defaults or copies of these numbers in code, and do not add new config files when one of
these fits. Steering limits must also match `../mdp_stm32/include/servo.h`.

## Conventions

- SI units and Nav2 parameter names internally. The frame chain is `map → odom → base_footprint → base_link`;
  `base_link` = middle of the **rear axle**.
- Anything a human reads or types (logs, tool arguments) uses **tablet cells**: col,row `0–19`,
  10 cm, `(0,0)` bottom-left, plus `N/E/S/W`. Never metres/cm there (`markers.cell()`,
  `markers.direction()`).
- Log lines: short, one per event, fixed-width tag (`LEG 1/4   -> #1 (5,8)E  fwd REV fwd`). Use the
  node's logger (→ `/rosout`), not custom log topics.
- Node `main()` = `mdp_bringup.utils.run.run(NodeClass)`; periodic loops = `utils.run.wall_timer`
  (a sim-time timer sometimes never started).
- Match the surrounding code's comment style: say *why*, with dates for measurements.

## Running

| Goal | Command |
| --- | --- |
| Sim, task 1 (obstacles from `tasks.yaml`) | `pixi run sim task:=1` then `pixi run reset`, `pixi run go` |
| Sim, task 2 | `pixi run sim task:=2` then `pixi run go` |
| Bare car (manual / calibration) | `pixi run sim` then `pixi run calib straight 1.0` / `rotate 90` / `turn left` / `goto 5 8 E` / `ultrasonic 60` |
| Real car | `pixi run real [task:=1\|2]` on the Pi |
| Headless | add `gui:=false`; no camera: `vision:=false` |

Launch arguments, tablet protocol and pixi tasks: `../docs/quickstart.md`.

### Test sims must be isolated

The user often has their own sim or Foxglove running. `pixi.toml` forces `ROS_DOMAIN_ID=0`, so
override it **inside** pixi, and give Gazebo its own partition:

```bash
env GZ_PARTITION=agent_test pixi run -- env ROS_DOMAIN_ID=77 \
  ros2 launch mdp_bringup mdp.launch.py sim:=true task:=1 gui:=false bluetooth_device:=/tmp/no_tablet
```

Run every other command for that sim the same way (`env GZ_PARTITION=agent_test pixi run -- env
ROS_DOMAIN_ID=77 ros2 run mdp_bringup trigger /start_run`). To stop it, signal only processes whose
environment has `ROS_DOMAIN_ID=77` / `GZ_PARTITION=agent_test`. **Never `pkill ros2`/`gz`**: that kills
the user's sim.

### Checking a change actually works

1. Tests pass.
2. A sim run finishes: task 1 logs `FINISHED  all obstacles visited`, task 2
   `FINISHED  in the carpark`, with no `Traceback` in the launch output.
3. For driving changes, compare with Gazebo's truth: record a bag (exclude images) and read
   `/sim/ground_truth` (`transforms[0]` = the car) against `/run_status.x/y` or `/odometry/filtered`.
   Reference results (task 1): stops 1–6 cm off the checkpoint, closest body gap to a block ≥ ~1 cm,
   YOLO reads 4/4.

## Real-car safety

- Driving commands (`go`, `calib`, `teleop`, `around-obstacle`) move a real car when `pixi run real` is
  up. Don't run them unless the user asks. For tests, the wheels should be off the ground.
- `calib` refuses to run while a task runner owns `/cmd_vel`; keep that check in any new driving tool.
- Flashing the STM32 (`../mdp_stm32`, `pixi run flash`) only when asked.

## Git

- Commit or push only when the user asks. Branch `main`; `mdp_ros` is a submodule of `../` (`mdp`):
  after committing here, commit the pointer bump in the parent.
- **No `Co-Authored-By` or other AI attribution** in commit messages.
- Don't commit `build/ install/ log/ bags/`, or files the user didn't ask for; ask about untracked
  files you didn't create.

## Working with the user

- Explain proposed changes (what and why) and wait for a yes before restructuring or deleting their
  code. Small fixes inside the task are fine.
- When sim and real, or two options, differ: list the differences and let the user choose.
- Answers as short tables / diagrams, not long paragraphs.
