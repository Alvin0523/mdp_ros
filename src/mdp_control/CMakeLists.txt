cmake_minimum_required(VERSION 3.8)
project(mdp_control)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/reeds_shepp_planner.py
    scripts/spline_planner.py
    scripts/pure_pursuit_follower.py
    scripts/task1_runner.py
    scripts/task2_runner.py
    scripts/yolo_arrow_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
