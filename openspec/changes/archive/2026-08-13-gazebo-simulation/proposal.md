## Why

The `mdp_ros` workspace currently has URDF robot descriptions (`mdp_description`) and bringup launches (`mdp_bringup`), but lacks Gazebo simulation integration. Enabling Gazebo simulation allows testing Ackermann steering controls, sensor pipelines, and robot navigation in a risk-free virtual environment before deploying to physical STM32 and ROS2 hardware.

## What Changes

- Add Gazebo ROS2 integration (`ros_gz`, `gz_ros2_control` / `ros2_control` plugin configuration) to the `mini_akm` URDF description.
- Add Gazebo simulation launch configurations (`sim.launch.py`) in `mdp_bringup` to launch Gazebo Harmonic / Jazzy simulation, spawn the `mini_akm` robot model, and activate joint state/ackermann controllers.
- Configure `ackermann_steering_controller` and `joint_state_broadcaster` controllers for Gazebo simulation environment.
- Add bridge nodes/configs for Gazebo topics (clock, tf, cmd_vel, joint states, sensors).

## Capabilities

### New Capabilities
- `gazebo-simulation`: Spawning the `mini_akm` Ackermann-steered robot in Gazebo, exposing `/cmd_vel` velocity control, and publishing TF / joint states in simulation.

### Modified Capabilities
<!-- None -->

## Impact

- `mdp_description`: URDF update to include `<gazebo>` tags, `gz_ros2_control` hardware interface plugin reference, and physics/transmission definitions.
- `mdp_bringup`: New `sim.launch.py` and controller parameter YAML (`config/ackermann_controller.yaml`).
- Dependencies: Requires `ros_gz_sim`, `gz_ros2_control`, and `ackermann_steering_controller` ROS2 Jazzy packages.
