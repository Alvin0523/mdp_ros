# bringup Specification

## Purpose
Provides the workspace's single top-level launch entrypoint, `bringup.launch.py`, which composes the robot description's display launch into a runnable robot bringup.
## Requirements
### Requirement: Top-level bringup launch
The `mdp_bringup` package SHALL provide a `bringup.launch.py` launch file that includes `mdp_description`'s `display.launch.py`, with the joint-state GUI disabled, as the workspace's top-level entrypoint.

#### Scenario: Launching bringup with defaults
- **WHEN** `bringup.launch.py` is launched with no arguments
- **THEN** `mdp_description`'s `display.launch.py` is included with `use_joint_state_gui:=false`
- **AND** `robot_state_publisher` and `joint_state_publisher` start
- **AND** `rviz2` starts (since `use_rviz` defaults to `true`)

### Requirement: Bringup RViz is toggleable
The `bringup.launch.py` launch file SHALL expose a `use_rviz` launch argument that controls whether RViz is started.

#### Scenario: Disabling RViz at bringup
- **WHEN** `bringup.launch.py` is launched with `use_rviz:=false`
- **THEN** `rviz2` is not started

