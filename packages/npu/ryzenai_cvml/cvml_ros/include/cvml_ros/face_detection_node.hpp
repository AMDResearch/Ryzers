// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef RYZEN_AI_CVML__FACE_DETECTION_NODE_HPP_
#define RYZEN_AI_CVML__FACE_DETECTION_NODE_HPP_

#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "cvml-face-detector.h"
#include "cvml-context.h"

namespace cvml_ros
{

class FaceDetectionNode : public rclcpp::Node
{
public:
  explicit FaceDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaceDetectionNode();

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  // CVML objects
  amd::cvml::Context* context_;
  std::unique_ptr<amd::cvml::FaceDetector> face_detector_;

  // ROS2 objects
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_pub_;

  // Parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string fd_model_type_;
};

}  // namespace cvml_ros

#endif  // RYZEN_AI_CVML__FACE_DETECTION_NODE_HPP_
