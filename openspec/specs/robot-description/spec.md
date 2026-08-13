# robot-description Specification

## Purpose
Provides the URDF model, meshes, and RViz configuration for the mdp_car robot (`mini_akm_robot`), plus a `display.launch.py` that publishes robot state and optionally visualizes it in RViz.
## Requirements
### Requirement: URDF model availability
The `mdp_description` package SHALL provide a URDF file describing the `mini_akm_robot` robot, along with the mesh files it references, installed to the package's share directory so other packages can locate them via `package://mdp_description/...` URIs.

#### Scenario: URDF resolves its mesh references
- **WHEN** the URDF is loaded (e.g. via `xacro` and `robot_state_publisher`)
- **THEN** every `package://mdp_description/meshes/...` reference in the URDF resolves to an installed mesh file under the package's share directory

### Requirement: Display launch publishes robot state
The package SHALL provide a `display.launch.py` launch file that starts `robot_state_publisher` with the robot description generated from the URDF via `xacro`.

#### Scenario: Launching with defaults
- **WHEN** `display.launch.py` is launched with no arguments
- **THEN** `robot_state_publisher` starts and publishes the `robot_description` parameter derived from `mini_akm_robot.urdf`
- **AND** `joint_state_publisher_gui` starts (since `use_joint_state_gui` defaults to `true`)
- **AND** `rviz2` starts using the package's default RViz config (since `use_rviz` defaults to `true`)

### Requirement: Display launch is configurable
The `display.launch.py` launch file SHALL expose launch arguments to select between `joint_state_publisher` and `joint_state_publisher_gui`, to enable or disable RViz, and to override the URDF path and RViz config path.

#### Scenario: Disabling the joint state GUI
- **WHEN** `display.launch.py` is launched with `use_joint_state_gui:=false`
- **THEN** `joint_state_publisher` starts instead of `joint_state_publisher_gui`

#### Scenario: Disabling RViz
- **WHEN** `display.launch.py` is launched with `use_rviz:=false`
- **THEN** `rviz2` is not started

