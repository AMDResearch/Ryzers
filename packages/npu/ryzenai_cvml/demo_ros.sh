#!/bin/bash

source /opt/ros/kilted/setup.bash
colcon build --packages-select cvml_ros

# Launch webcam node
ros2 run v4l2_camera v4l2_camera_node &

# Launch web server
ros2 run web_video_server web_video_server &

# Launch cvml nodes
source install/setup.sh
ros2 run cvml_ros face_mesh_node --ros-args -p input_topic:=/image_raw &
ros2 run cvml_ros depth_estimation_node --ros-args -p input_topic:=/image_raw &
ros2 run cvml_ros face_detection_node --ros-args -p input_topic:=/image_raw 
