# Ryzen AI Depth Estimation ROS2 Node

A ROS2 node that uses AMD's Ryzen AI CVML Library for real-time depth estimation.

## Features

- Real-time depth estimation using Ryzen AI hardware acceleration
- CPU backend support (configurable for GPU/NPU)
- Standard ROS2 image transport interface
- Configurable input/output topics

## Dependencies

- ROS2 (tested with Humble/Iron)
- OpenCV
- cv_bridge
- image_transport
- Ryzen AI CVML Library

## Building

1. Ensure the Ryzen AI CVML Library is properly installed
2. Set OPENCV_INSTALL_ROOT environment variable
3. Build with colcon:

```bash
cd /path/to/ros2_workspace
colcon build --packages-select ryzen_ai_depth_estimation
```

## Usage

```
colcon build --packages-select ryzen_ai_depth_estimation --cmake-clean-cache
source install/setup.bash

ros2 run cvml_ros video_publisher.py \
    --ros-args \
    -p video_path:=/ryzers/RyzenAI-SW/Ryzen-AI-CVML-Library/samples/video_call.mp4 \
    -p topic:=/camera/image_raw

ros2 run cvml_ros depth_estimation_node
```



### Launch the node:

```bash
ros2 launch ryzen_ai_depth_estimation depth_estimation.launch.py
```

### With custom topics:

```bash
ros2 launch ryzen_ai_depth_estimation depth_estimation.launch.py \
  input_topic:=/my_camera/image_raw \
  output_topic:=/my_depth/depth
```

### Run directly:

```bash
ros2 run ryzen_ai_depth_estimation depth_estimation_node
```

## Topics

- **Input**: `/camera/image_raw` (sensor_msgs/Image) - RGB input images
- **Output**: `/depth_estimation/depth` (sensor_msgs/Image) - Float32 depth maps

## Parameters

- `input_topic` (string, default: "/camera/image_raw"): Input image topic
- `output_topic` (string, default: "/depth_estimation/depth"): Output depth topic

## Notes

- The node is configured to use CPU backend by default
- For NPU/GPU backends, modify the `SetInferenceBackend()` call in the source code
- Input images are expected in RGB format
- Output depth maps are in float32 format with relative depth values
