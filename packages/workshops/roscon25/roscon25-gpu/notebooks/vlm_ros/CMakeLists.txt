cmake_minimum_required(VERSION 3.10)
project(vlm_ros)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)

# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/VLMQuery.srv"
  DEPENDENCIES sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install Python modules (guard against duplicate target definition when re-configuring)
if(NOT TARGET ament_cmake_python_symlink_${PROJECT_NAME}
   AND NOT TARGET ament_cmake_python_copy_${PROJECT_NAME})
  ament_python_install_package(${PROJECT_NAME})
endif()

# Install Python executables
install(PROGRAMS
  vlm_ros/vlm_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vlm_node
)

ament_package()
