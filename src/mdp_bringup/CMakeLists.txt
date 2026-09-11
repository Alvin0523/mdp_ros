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
  scripts/go.py
  scripts/drive_distance.py
  # generate_arena_world.py removed 2026-09-11 - obstacles now live directly in
  # mdp_description/worlds/task1_arena.sdf, which the sim launches load as-is.
  # Not an entry point: a pure helper module the runners import as a sibling out
  # of lib/mdp_bringup, installed here so that import resolves at runtime.
  scripts/pose_transform.py
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
