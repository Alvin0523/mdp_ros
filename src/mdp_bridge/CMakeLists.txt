cmake_minimum_required(VERSION 3.8...3.28)
project(mdp_bridge)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(ament_dependencies
  rclcpp
  sensor_msgs
  std_msgs
)

include_directories(
  include
)

add_executable(serial_bridge_node src/serial_bridge_node.cpp)
ament_target_dependencies(serial_bridge_node ${ament_dependencies})

# bluetooth_bridge_node talks to BlueZ's kernel RFCOMM socket API directly
# (bind/listen/accept a server socket) rather than a pre-bound /dev/rfcommN
# device file, so it needs libbluetooth's headers/lib - install with
# `sudo apt install libbluetooth-dev` on the Pi if this fails to find it.
find_library(BLUEZ_LIBRARY NAMES bluetooth)
if(NOT BLUEZ_LIBRARY)
  message(FATAL_ERROR
    "libbluetooth not found - install libbluetooth-dev (BlueZ) to build bluetooth_bridge_node")
endif()

add_executable(bluetooth_bridge_node src/bluetooth_bridge_node.cpp)
ament_target_dependencies(bluetooth_bridge_node ${ament_dependencies})
target_link_libraries(bluetooth_bridge_node ${BLUEZ_LIBRARY})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  serial_bridge_node
  bluetooth_bridge_node
  DESTINATION lib/${PROJECT_NAME})
ament_package()
