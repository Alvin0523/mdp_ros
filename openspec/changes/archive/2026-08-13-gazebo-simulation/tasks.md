## 1. URDF & Controller Configuration

- [x] 1.1 Add `<gazebo>` tags, link/joint inertia/friction settings, and `gz_ros2_control` hardware interface plugin configuration to `mdp_description/urdf/mini_akm_robot.urdf`.
- [x] 1.2 Create `mdp_bringup/config/ackermann_controller.yaml` configuring `joint_state_broadcaster` and `ackermann_steering_controller` parameters for steering/drive joints.

## 2. Launch & Bridge Integration

- [x] 2.1 Create `mdp_bringup/launch/sim.launch.py` to start Gazebo Harmonic/Jazzy simulation environment, load robot model, and spawn controller manager and controllers.
- [x] 2.2 Configure `ros_gz_bridge` in `sim.launch.py` to bridge `/clock`, `/cmd_vel`, `/joint_states`, and TF topics between ROS2 and Gazebo.

## 3. Verification & Testing

- [x] 3.1 Verify building workspace (`pixi run build`) with new launch and configuration files.
- [x] 3.2 Test launching Gazebo simulation (`ros2 launch mdp_bringup sim.launch.py`) and verify model rendering and joint state publishing.
- [x] 3.3 Test teleop drive control on `/cmd_vel` using `teleop_twist_keyboard` to verify Ackermann movement in Gazebo.
