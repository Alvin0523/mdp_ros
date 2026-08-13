# Proposal: EKF Sensor Fusion (Wheel Odometry + IMU)

## Summary
Add IMU-fused odometry for task2 via `robot_localization`'s EKF, sitting downstream of `ackermann_steering_controller`. Tested first in simulation, using a simulated IMU sensor in Gazebo, so the exact same node graph and `ekf.yaml` shape will later apply unchanged to real hardware.

## Motivation & Context
`task2_runner.py` currently reads `odom→base_footprint` TF and treats it as "ground truth" (see the file's own docstring: "True Odometry & Ground Truth Position-Reaching"). It isn't — that TF is broadcast by `ackermann_steering_controller` itself, the same wheel-encoder dead-reckoning as `/ackermann_steering_controller/odometry`, just read a second way. There is no real ground truth anywhere in the current pipeline.

Wheel-only odometry is specifically weakest at heading (yaw) under wheel slip during sharp turns — exactly what task2's slalom around two obstacles demands. This is a real, independent contributor to the obstacle-clearance problems worked through earlier this session (separate from the path-planner fixes already made in `spline_planner.py`/`task2_runner.py`).

Per **ADR 0003** (Pattern B: kinematics/odometry live on the host, not the MCU; IMU bypasses `ros2_control` entirely and is published as a plain topic), fusing that IMU topic with wheel odometry via `robot_localization` is the natural next step of an already-accepted architecture decision, not a new direction.

Real hardware cannot exercise either EKF input yet: per `docs/architecture.md`'s own status table, `mdp_stm32` (separate Zephyr firmware repo) has encoder reading, IMU reading, and micro-ROS integration all listed as stubs / not started. This proposal scopes to what's testable now — sim — and prepares (but does not activate) the real-hardware side.

## Proposed Changes

### `mdp_ros`
- **Simulated IMU**: add an `imu` sensor to `mini_akm_robot.urdf` (`gz-sim-imu-system`, nested per gz-sim's own IMU example convention), publishing `/imu/data`. Attached to `base_link` directly (not a dedicated `imu_link`) as a deliberate simplification.
- **Dependency**: add `ros-jazzy-robot-localization` to `pixi.toml`.
- **EKF config**: new `src/mdp_bringup/config/ekf.yaml` (sim) fusing `/ackermann_steering_controller/odometry` (linear x only) with `/imu/data` (yaw + yaw rate), `base_link_frame: base_footprint` to match the sim URDF's TF root. New `src/mdp_bringup/config/ekf_real.yaml` (real, prep-only, not launched yet) — identical except `base_link_frame: base_link` (real URDF has no `base_footprint`).
- **TF ownership**: set `enable_odom_tf: false` on `ackermann_steering_controller` (sim config) so the EKF becomes the sole publisher of `odom→base_footprint`, avoiding two nodes competing for the same transform.
- **Launch wiring**: add an `ekf_node` to `task2_sim.launch.py`, plus the `/imu/data` bridge line in the existing `ros_gz_bridge parameter_bridge`. `real.launch.py` is *not* updated to launch `ekf_node` yet — nothing real exists for it to consume.
- **Honesty fix**: rename `task2_runner.py`'s `self.gt_x/gt_y/gt_yaw`/`update_ground_truth_tf()` to reflect what they actually are (a filtered estimate, not ground truth), and correct the file's docstring.

### `docs`
- New `docs/adr/0005-ekf-sensor-fusion.md` (+ row in `docs/adr/index.md`).
- `docs/architecture.md`: stack table, both diagrams, and the "What's implemented vs. TODO" status table updated — including an explicit note that real-hardware activation is blocked on `mdp_stm32` firmware.
- `docs/ros/simulation.md`: new section documenting the IMU + EKF setup and the resulting TF ownership change.

### Explicitly out of scope
- `mdp_stm32` firmware (encoder/IMU/micro-ROS) — separate repo, the actual blocker for real-hardware activation.
- A literal Gazebo ground-truth bridge (bridging true simulated pose) for debugging/comparison — not needed for this change; can be proposed separately if wanted.

## Verification Plan
1. `pixi install` (pulls in `robot_localization`), `pixi run build`.
2. `pixi run sim-task2`; confirm `/imu/data` is publishing and `/odometry/filtered` is produced by the EKF.
3. `ros2 run tf2_ros tf2_echo odom base_footprint` — confirm exactly one TF publisher (no "multiple authorities for frame" warning) after `enable_odom_tf: false`.
4. Foxglove: TF tree still connects end to end; robot's estimated pose visibly tracks better through the sharp slalom turns than the wheel-only estimate did.
5. Re-run the task2 slalom; obstacle clearance holds up at least as well as the offline-simulated numbers from the path-planner work.
