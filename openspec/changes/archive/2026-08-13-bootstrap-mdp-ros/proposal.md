## Why

This workspace (`mdp_ros`) is a fresh ROS 2 workspace with an empty `src/` tree. A working `mdp_description`/`mdp_bringup` pair already exists at `references/mdp_ws/src/`: the URDF (`mini_akm_robot.urdf`) is a SolidWorks CAD export of this specific robot's actual chassis and steering geometry, not a generic vendor demo model. Porting it gives this workspace a visualizable, dimensionally-accurate robot model and a top-level launch entrypoint without re-deriving them from scratch.

## What Changes

- Add `mdp_description` package: URDF (`mini_akm_robot.urdf`, a SolidWorks-exported CAD model of the robot's chassis and Ackermann steering links), STL meshes, RViz config, and a `display.launch.py` that starts `robot_state_publisher`, `joint_state_publisher`/`joint_state_publisher_gui`, and `rviz2`.
- Add `mdp_bringup` package: `bringup.launch.py` that includes `mdp_description`'s display launch (with the joint-state GUI disabled) as the workspace's top-level entrypoint.
- Both packages are ported as `ament_cmake` packages with `CMakeLists.txt` and `package.xml`, matching the source package layout, and installing their `launch`/`urdf`/`meshes`/`rviz` directories to `share/<pkg>`.
- Update package maintainer/version metadata only if it conflicts with this workspace's conventions; otherwise carry over unchanged from the source.

## Capabilities

### New Capabilities
- `robot-description`: URDF/mesh model of the mdp_car robot and a `display.launch.py` that publishes robot state and optionally opens RViz/joint_state_publisher_gui.
- `bringup`: Top-level `bringup.launch.py` that composes `robot-description`'s display launch as the workspace's single entrypoint.

### Modified Capabilities
(none — this workspace currently has no packages or specs)

## Impact

- Adds two new ROS 2 packages under `src/`: `mdp_description`, `mdp_bringup`.
- New runtime dependencies (already present in `pixi.toml`): `robot_state_publisher`, `joint_state_publisher`, `joint_state_publisher_gui`, `rviz2`, `xacro`.
- No existing code, specs, or launch files are modified since the workspace is currently empty.