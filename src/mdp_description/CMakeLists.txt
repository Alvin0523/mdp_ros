cmake_minimum_required(VERSION 3.8...3.28)
project(mdp_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf meshes rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
