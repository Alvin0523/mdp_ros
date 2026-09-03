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
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
