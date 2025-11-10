cmake_minimum_required(VERSION 3.8)
project(cvml_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-pedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Ryzen AI CVML Library
set(AMD_CVML_SDK_ROOT /ryzers/RyzenAI-SW/Ryzen-AI-CVML-Library)
find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

# Include directories
include_directories(include)
include_directories(${AMD_CVML_SDK_ROOT}/include)
include_directories(${cv_bridge_INCLUDE_DIRS})

# Create depth estimation executable
add_executable(depth_estimation_node
  src/depth_estimation_main.cpp
  src/depth_estimation_node.cpp
)

target_link_libraries(depth_estimation_node
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  cv_bridge::cv_bridge
  image_transport::image_transport
  ${std_msgs_TARGETS}
  ${OpenCV_LIBRARIES}
  ${RyzenAILibrary_LIBS}
)

# Create face mesh executable
add_executable(face_mesh_node
  src/face_mesh_main.cpp
  src/face_mesh_node.cpp
)

target_link_libraries(face_mesh_node
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  cv_bridge::cv_bridge
  image_transport::image_transport
  ${std_msgs_TARGETS}
  ${OpenCV_LIBRARIES}
  ${RyzenAILibrary_LIBS}
)

# Create face detection executable
add_executable(face_detection_node
  src/face_detection_main.cpp
  src/face_detection_node.cpp
)

target_link_libraries(face_detection_node
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  cv_bridge::cv_bridge
  image_transport::image_transport
  ${std_msgs_TARGETS}
  ${OpenCV_LIBRARIES}
  ${RyzenAILibrary_LIBS}
)

# Install executables
install(TARGETS
  depth_estimation_node
  face_mesh_node
  face_detection_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
	scripts/video_publisher.py
	DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
