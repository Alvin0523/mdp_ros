## 1. Port mdp_description

- [ ] 1.1 Create `src/mdp_description/` and copy `package.xml`, `CMakeLists.txt` from `references/mdp_ws/src/mdp_description/`
- [ ] 1.2 Copy `urdf/mini_akm_robot.urdf` into `src/mdp_description/urdf/`
- [ ] 1.3 Copy `meshes/mini_akm_robot_meshes/*.STL` into `src/mdp_description/meshes/mini_akm_robot_meshes/`
- [ ] 1.4 Copy `rviz/display.rviz` into `src/mdp_description/rviz/`
- [ ] 1.5 Copy `launch/display.launch.py` into `src/mdp_description/launch/`

## 2. Port mdp_bringup

- [ ] 2.1 Create `src/mdp_bringup/` and copy `package.xml`, `CMakeLists.txt` from `references/mdp_ws/src/mdp_bringup/`
- [ ] 2.2 Copy `launch/bringup.launch.py` into `src/mdp_bringup/launch/`

## 3. Verify

- [ ] 3.1 Build the workspace with `pixi run build` and confirm both `mdp_description` and `mdp_bringup` build without errors
- [ ] 3.2 Launch `ros2 launch mdp_description display.launch.py use_rviz:=false` and confirm `robot_description` is published with no missing mesh errors
- [ ] 3.3 Launch `ros2 launch mdp_bringup bringup.launch.py use_rviz:=false` and confirm `robot_state_publisher` and `joint_state_publisher` (not the GUI) start
- [ ] 3.4 Run `openspec validate bootstrap-mdp-ros --strict` and fix any reported issues