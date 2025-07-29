cmake_minimum_required(VERSION 3.5)
project(vla_client)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vla_interfaces REQUIRED)

install(
  PROGRAMS # making it executable
    nodes/vla_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vla_bridge_node
)

ament_package()