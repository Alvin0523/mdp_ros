cmake_minimum_required(VERSION 3.8...3.28)
project(mdp_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/task1_runner.py
  scripts/task2_runner.py
  scripts/publish_test_obstacles.py
  scripts/trigger.py
  scripts/drive_distance.py
  scripts/rotate_angle.py
  scripts/circle_obstacle.py
  scripts/bt_monitor.py
  scripts/manual_drive.py
  scripts/send_target.py
  scripts/robot_pose_feedback.py
  scripts/grid_guard.py  # TEMP: zero /cmd_vel when the robot leaves the tablet grid
  # Not entry points: pure helper modules imported as siblings out of
  # lib/mdp_bringup (obstacle_layout also by launch/mdp.launch.py), installed
  # here so those imports resolve at runtime.
  scripts/pose_transform.py
  scripts/obstacle_layout.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # mdp_bringup is ament_cmake, so pytest directories are NOT auto-discovered
  # the way they are for ament_python packages - this registers test/ explicitly
  # so `colcon test` (pixi run test) picks it up.
  ament_add_pytest_test(mdp_bringup_pytest test
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_package()
