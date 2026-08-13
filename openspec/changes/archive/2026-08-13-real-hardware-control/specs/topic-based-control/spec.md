# Delta Spec: Real Hardware Control

## ADDED REQUIREMENTS

### Requirement: Topic-Based Hardware Interface
The host ROS2 system SHALL provide a hardware bringup launch (`real.launch.py`) that instantiates `topic_based_ros2_control/TopicBasedSystem` hardware interface for real hardware operation.

#### Scenario: Real Hardware Control Launch
- **GIVEN** a physical WHEELTEC robot connected via USB serial (`/dev/ttyUSB0`)
- **WHEN** `ros2 launch mdp_bringup real.launch.py` is executed
- **THEN** `controller_manager` SHALL activate `ackermann_steering_controller` and `joint_state_broadcaster` using `topic_based_ros2_control`
- **AND** `ackermann_steering_controller` SHALL publish joint targets to `/joint_commands` and read feedback from `/joint_states`.

### Requirement: micro-ROS Agent Transport Node
The host ROS2 system SHALL provide a micro-ROS Agent node launch configuration to communicate over serial UART with the STM32 MCU.

#### Scenario: Serial Transport Link
- **GIVEN** the micro-ROS Agent node is started
- **WHEN** commands are received on `/joint_commands`
- **THEN** micro-ROS Agent SHALL bridge the topic messages over serial UART (`/dev/ttyUSB0` @ 115200 baud) to the STM32 MCU.
