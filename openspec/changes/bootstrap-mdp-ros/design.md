## Context

`mdp_ros`'s `src/` is currently empty. `mdp_description` and `mdp_bringup` already exist at `references/mdp_ws/src/{mdp_description,mdp_bringup}` as plain `ament_cmake` packages (URDF + meshes + RViz config + launch files, installed via `CMakeLists.txt`) — this is where `mdp_description`'s CAD-exported URDF (`mini_akm_robot.urdf`, a SolidWorks export of this robot's actual chassis and steering geometry) currently lives; it was not derived from WHEELTEC's generic vendor tutorial packages under `references/1.*`–`4.*`, which model a different, simplified chassis for their own ROS1 demos. This workspace's `pixi.toml` already lists the runtime deps these packages need (`xacro`, `robot_state_publisher`, `joint_state_publisher`, `joint_state_publisher_gui`, `rviz2`). See proposal.md for motivation; specs/robot-description and specs/bringup for required behavior.

## Goals / Non-Goals

**Goals:**
- Copy `mdp_description` and `mdp_bringup` into this workspace's `src/` with no functional changes to URDF, meshes, RViz config, or launch behavior.
- Keep both packages buildable with `colcon build --symlink-install` (the workspace's existing build task) under `ros-jazzy` / Jazzy.

**Non-Goals:**
- No conversion of the URDF to xacro macros, no mesh re-export, no RViz config redesign.
- No changes to `references/mdp_ws` itself (read-only source).
- No addition of ros2_control / Gazebo integration here, even though `pixi.toml` has those deps staged for future work — out of scope for this port.

## Decisions

- **Straight copy, not re-derivation**: Copy the package directories verbatim (URDF, STL meshes, RViz config, launch files, `CMakeLists.txt`, `package.xml`) rather than rewriting them, since the CAD-exported model is already correct and the goal is parity, not redesign. Alternative considered: regenerate URDF from CAD, or start from WHEELTEC's generic vendor URDF — both rejected: the former is unnecessary scope for a port, the latter would replace this robot's actual chassis geometry with a different, generic one.
- **Keep `ament_cmake` build type**: Preserve the reference packages' `ament_cmake` + `CMakeLists.txt install(DIRECTORY ...)` pattern rather than converting to `ament_python`, since there's no Python code in either package and the install-directory pattern is simpler for static assets (URDF/meshes/launch/rviz).
- **package.xml metadata carried over as-is**: Maintainer email (`alvinwm0523@gmail.com`) already matches this workspace's `pixi.toml` author, so no metadata changes are needed. `license` stays `TODO`, matching the source.
- **No `pixi.toml` changes**: All exec-time dependencies (`xacro`, `robot_state_publisher`, `joint_state_publisher`, `joint_state_publisher_gui`, `rviz2`) are already present in `mdp_ros/pixi.toml`, so this change touches no dependency manifests.

## Risks / Trade-offs

- [STL meshes are binary and sizable] → Copy them unmodified via filesystem copy (not recreated), keeping the port a byte-for-byte asset transfer that's easy to diff against the source if needed.
- [Divergence from `references/mdp_ws` over time] → Out of scope to solve here; this is a one-time port, not a sync mechanism. Future CAD revisions would need a separate manual re-port.
- [No automated test currently verifies the launch files actually start the nodes] → Mitigate by building the workspace (`colcon build`) and doing a manual `ros2 launch mdp_bringup bringup.launch.py` smoke check as part of task verification.