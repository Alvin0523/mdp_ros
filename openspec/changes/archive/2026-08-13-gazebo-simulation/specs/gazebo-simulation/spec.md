## Purpose

Provides Gazebo Harmonic / Jazzy simulation capabilities for the `mini_akm` Ackermann-steered robot in `mdp_ros`, including ROS2 control interface, velocity command listening, joint state publishing, and TF broadcasting.

## ADDED Requirements

### Requirement: Gazebo Robot Model Spawning
The system SHALL support launching Gazebo simulation and spawning the `mini_akm` robot URDF model with physics and visual meshes loaded.

#### Scenario: Spawn robot in Gazebo world
- **WHEN** user launches `ros2 launch mdp_bringup sim.launch.py`
- **THEN** Gazebo simulation opens, the world loads, and the `mini_akm` robot is spawned at origin without physics explosion or missing mesh errors.

### Requirement: Gazebo Ackermann Steering Control
The system SHALL expose an `/cmd_vel` subscriber interface via `ackermann_steering_controller` in Gazebo to drive the simulated robot.

#### Scenario: Teleoperation in simulation
- **WHEN** velocity commands (`geometry_msgs/Twist`) are published to `/cmd_vel`
- **THEN** the simulated robot's steering joints angle and rear wheel joints rotate according to Ackermann kinematics in Gazebo.

### Requirement: Gazebo Joint State and TF Telemetry
The system SHALL publish `/joint_states` and maintain the robot `/tf` tree within the Gazebo simulation runtime environment.

#### Scenario: Joint states and TF published
- **WHEN** the Gazebo simulation is running
- **THEN** `/joint_states` topics are published by `joint_state_broadcaster` and `robot_state_publisher` updates TF frames for all robot links.
