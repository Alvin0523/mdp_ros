# Design: Real Hardware Control (`topic_based_ros2_control` + micro-ROS Agent Setup)

## Architectural Overview

```text
========================================================================================
                                     INPUT LAYER
                             [pixi run teleop] / [Nav2]
                                         │
                                         ▼ (/cmd_vel)
                               [ackermann_steering_controller] (ROS2 Host Controller)
========================================================================================
                                HARDWARE INTERFACE LAYER
                                         │
                                         ▼
                             topic_based_ros2_control
                             (TopicBasedSystem Plugin)
                                         │
                                         ├─► /joint_commands (sensor_msgs/JointState)
                                         └─◄ /joint_states   (sensor_msgs/JointState)
========================================================================================
                                   EXECUTION LAYER
                               [micro-ROS Agent Node]
                               (Host - Raspberry Pi 4B)
                                         │
                                         │ USB Serial UART (/dev/ttyUSB0 @ 115200)
                                         ▼
                                [STM32 MCU Firmware]
                                - Dual AT8236 Motor PWM (MG513P3012V)
                                - HWZ020 Steering Servo PWM
                                - Hall Encoders & ICM-20948 IMU
========================================================================================
```

## Detailed Specifications

### 1. `topic_based_ros2_control` Configuration
`TopicBasedSystem` acts as a standard C++ `hardware_interface::SystemInterface` plugin for `ros2_control`. It intercepts commands from `ackermann_steering_controller` and packages them into a ROS2 `sensor_msgs/msg/JointState` message published on `/joint_commands`.

- **Command Topic:** `/joint_commands` (`sensor_msgs/msg/JointState`)
  - `name`: `['left_joint', 'right_joint', 'lb_joint', 'rb_joint']`
  - `position`: Steering angles (rad) for `left_joint` & `right_joint`
  - `velocity`: Rear wheel speeds (rad/s) for `lb_joint` & `rb_joint`
- **State Topic:** `/joint_states` (`sensor_msgs/msg/JointState`)
  - Feeds actual hardware encoder wheel positions & speeds back into `joint_state_broadcaster`.

### 2. micro-ROS Agent Transport
The `micro_ros_agent` ROS2 node runs on the host (RPi 4B) and opens a serial transport session over `/dev/ttyUSB0` at `115200` baud (matching `USART3` on the WHEELTEC C30D board).
