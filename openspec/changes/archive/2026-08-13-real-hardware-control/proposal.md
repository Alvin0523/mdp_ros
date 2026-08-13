# Proposal: Real Hardware Control (`topic_based_ros2_control` Setup)

## Summary
Configure host-side ROS2 Control for real hardware operation using `topic_based_ros2_control`. This allows `ackermann_steering_controller` on the host (Raspberry Pi 4B) to communicate with the hardware layer via standard ROS2 topics (`/joint_commands` and `/joint_states`).

## Motivation & Context
Per **ADR 0002** (`topic_based_ros2_control`) and **ADR 0003** (Pattern B Kinematics), the exact same `ackermann_steering_controller` used in Gazebo simulation is reused on real hardware. 

Instead of writing a custom hardware interface or putting complex kinematics on the MCU:
1. `ackermann_steering_controller` outputs joint velocity & position targets.
2. `topic_based_ros2_control` (`TopicBasedSystem`) publishes joint commands onto `/joint_commands` (`sensor_msgs/msg/JointState`) and receives encoder feedback from `/joint_states`.
3. The micro-ROS Agent node bridges `/joint_commands` and `/joint_states` over serial UART (`/dev/ttyUSB0` @ 115200 baud) to the STM32 MCU.

## Proposed Changes

### `mdp_ros`
- **Dependencies (`pixi.toml`):**
  - Added `ros-jazzy-topic-based-ros2-control = "*"`
  - Add shortcut task `hardware = "ros2 launch mdp_bringup real.launch.py"`
- **Bringup Launch (`src/mdp_bringup/launch/real.launch.py`):**
  - Launch `robot_state_publisher` with hardware URDF (`mini_akm_robot.urdf` configured for `TopicBasedSystem`).
  - Launch `controller_manager` with `topic_based_ros2_control` configuration.
  - Spawn `joint_state_broadcaster` and `ackermann_steering_controller`.
- **Controller Configuration (`src/mdp_bringup/config/real_controller.yaml`):**
  - Configure `topic_based_ros2_control` parameters (`joint_commands_topic: /joint_commands`, `joint_states_topic: /joint_states`).

## Verification Plan
1. `pixi install` verified — `ros-jazzy-topic-based-ros2-control` 0.3.0 resolved cleanly.
2. Run `pixi run build` to verify launch files and configs compile cleanly.
3. Test dry-run launch of `pixi run hardware` to verify node graph and topic structure (`/joint_commands`, `/joint_states`).
