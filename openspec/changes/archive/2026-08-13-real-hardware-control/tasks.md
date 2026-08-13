# Implementation Tasks: Real Hardware Control

- [x] 1. Add `ros-jazzy-topic-based-ros2-control` to `mdp_ros/pixi.toml` dependencies and verify `pixi install` <!-- id: 0 -->
- [x] 2. Create `src/mdp_bringup/config/real_controller.yaml` for `topic_based_ros2_control` and `ackermann_steering_controller` <!-- id: 1 -->
- [x] 3. Create `src/mdp_description/urdf/mini_akm_real_robot.urdf` configured for `topic_based_ros2_control/TopicBasedSystem` <!-- id: 2 -->
- [x] 4. Create `src/mdp_bringup/launch/real.launch.py` to launch `topic_based_ros2_control` hardware bringup <!-- id: 3 -->
- [x] 5. Add `hardware` shortcut task to `mdp_ros/pixi.toml` (`pixi run hardware`) <!-- id: 4 -->
- [x] 6. Build workspace and verify node graph, `/joint_commands`, and `/joint_states` topics <!-- id: 5 -->
