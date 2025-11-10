// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "cvml_ros/depth_estimation_node.hpp"
#include <opencv2/opencv.hpp>

namespace cvml_ros
{

DepthEstimationNode::DepthEstimationNode(const rclcpp::NodeOptions & options)
: Node("depth_estimation_node", options)
{
  // Declare parameters
  this->declare_parameter("input_topic", "/camera/image_raw");
  this->declare_parameter("output_topic", "/depth_estimation/depth");
  input_topic_ = this->get_parameter("input_topic").as_string();
  output_topic_ = this->get_parameter("output_topic").as_string();
  
  RCLCPP_INFO(this->get_logger(), "Input topic: %s", input_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic_.c_str());
  
  try {
    // Create CVML context
    context_ = amd::cvml::CreateContext();
    if (!context_) {
      throw std::runtime_error("Failed to create CVML context");
    }
    
    // Set GPU backend
    context_->SetInferenceBackend(amd::cvml::Context::InferenceBackend::AUTO);
    
    // Set streaming mode for real-time processing
    context_->SetStreamingMode(amd::cvml::Context::StreamingMode::ONLINE_STREAMING);
    
    // Create depth estimation feature
    depth_estimation_ = std::make_unique<amd::cvml::DepthEstimation>(context_);
    
    RCLCPP_INFO(this->get_logger(), "Ryzen AI Depth Estimation initialized");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize Ryzen AI: %s", e.what());
    throw;
  }
  
  // Create regular ROS2 publisher (getting weird weak_ptr issues with image_transport,
  // will keep simple publishers subscribers for now)
  depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);
  RCLCPP_INFO(this->get_logger(), "Created publisher on: %s", output_topic_.c_str());
  
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    input_topic_, 10,
    std::bind(&DepthEstimationNode::imageCallback, this, std::placeholders::_1));
  
  RCLCPP_INFO(this->get_logger(), "Created subscriber on: %s", input_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Depth estimation node started successfully");
}

DepthEstimationNode::~DepthEstimationNode()
{
  if (context_) {
    context_->Release();
  }
}

void DepthEstimationNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    // Convert ROS2 image to OpenCV
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat input_image = cv_ptr->image;
    
    RCLCPP_INFO_ONCE(this->get_logger(), "Received first image: %dx%d", input_image.cols, input_image.rows);
    
    // Create CVML input image
    amd::cvml::Image input_frame(
      amd::cvml::Image::Format::kRGB,
      amd::cvml::Image::DataType::kUint8,
      input_image.cols,
      input_image.rows,
      input_image.data
    );
    
    // Create CVML output image for depth map
    amd::cvml::Image output_img(
      amd::cvml::Image::Format::kGrayScale,
      amd::cvml::Image::DataType::kFloat32,
      input_image.cols,
      input_image.rows,
      nullptr
    );
    
    // Generate depth map
    bool success = depth_estimation_->GenerateDepthMap(input_frame, &output_img);
    if (!success) {
      RCLCPP_WARN(this->get_logger(), "Failed to generate depth map");
      return;
    }
    
    // Convert CVML output to OpenCV Mat
    float* depth_data = reinterpret_cast<float*>(output_img.GetBuffer());
    cv::Mat depth_map(
      static_cast<int>(output_img.GetHeight()),
      static_cast<int>(output_img.GetWidth()),
      CV_32FC1,
      depth_data
    );
    
    // Convert to ROS2 message
    std_msgs::msg::Header header;
    header.stamp = msg->header.stamp;
    header.frame_id = msg->header.frame_id;
    
    sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(
      header,
      sensor_msgs::image_encodings::TYPE_32FC1,
      depth_map
    ).toImageMsg();
    
    // Publish depth map
    depth_pub_->publish(*depth_msg);
    RCLCPP_INFO_ONCE(this->get_logger(), "Published first depth map");
    
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error processing image: %s", e.what());
  }
}

}  // namespace cvml_ros
