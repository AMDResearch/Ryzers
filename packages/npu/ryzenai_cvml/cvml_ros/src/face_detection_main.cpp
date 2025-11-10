// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "rclcpp/rclcpp.hpp"
#include "cvml_ros/face_detection_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<cvml_ros::FaceDetectionNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception in main: %s", e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
