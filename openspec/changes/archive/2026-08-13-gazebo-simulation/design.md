## Context

See `proposal.md` for motivation and system context. `mdp_description` contains the `mini_akm_robot.urdf` description ported from `references/mdp_ws/`. Per ADR 0003, both simulation and real hardware use `ackermann_steering_controller`. For Gazebo, `gz_ros2_control` is used to load the controllers.

## Goals / Non-Goals

**Goals:**
- Add `<gazebo>` tags and `gz_ros2_control` hardware interface plugin block to URDF.
- Create controller configuration `ackermann_controller.yaml` defining `joint_state_broadcaster` and `ackermann_steering_controller`.
- Create `sim.launch.py` in `mdp_bringup` to launch Gazebo (via `ros_gz_sim`), spawn the `mini_akm` model, load controller manager, and spawn controllers.
- Bridge Gazebo topics (clock, tf, cmd_vel, joint_states) using `ros_gz_bridge`.

**Non-Goals:**
- Hardware control via `topic_based_ros2_control` (this is Step 3).
- Custom Gazebo world plugins or complex obstacle environments (use standard empty or default world initially).

## Decisions

### Decision 1: Use `gz_ros2_control` plugin tag in URDF
- **Choice**: Integrate `<plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">` into `mini_akm_robot.urdf`.
- **Rationale**: standard ROS2 Jazzy + Gazebo Harmonic control bridge. Allows exact reuse of `ackermann_steering_controller`.

### Decision 2: Topic Bridge configuration
- **Choice**: Use `ros_gz_bridge` node in `sim.launch.py` to bridge `/clock` (`rosgraph_msgs/msg/Clock`), `/cmd_vel` (`geometry_msgs/msg/Twist`), and joint state/TF topics.
- **Rationale**: Ensures ROS2 time synchronization (`use_sim_time: true`) and seamless communication between ROS nodes and Gazebo.

## Risks / Trade-offs

- [Mesh path resolution in Gazebo] → Mitigation: Ensure package URI `package://mdp_description/meshes/...` is registered in `GZ_SIM_RESOURCE_PATH` or `AMENT_PREFIX_PATH` in launch environment.
- [Joint friction/physics tuning] → Mitigation: Standard joint parameters in URDF will be validated; damping/friction can be tuned if the robot slips or drifts unexpectedly.
