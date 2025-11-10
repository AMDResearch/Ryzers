#ifndef RYZEN_AI_CVML__DEPTH_ESTIMATION_NODE_HPP_
#define RYZEN_AI_CVML__DEPTH_ESTIMATION_NODE_HPP_

#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "cvml-depth-estimation.h"
#include "cvml-context.h"

namespace cvml_ros
{

class DepthEstimationNode : public rclcpp::Node
{
public:
  explicit DepthEstimationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~DepthEstimationNode();

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  
  // CVML objects
  amd::cvml::Context* context_;
  std::unique_ptr<amd::cvml::DepthEstimation> depth_estimation_;
  
  // ROS2 objects - regular publishers, NO image_transport
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  
  // Parameters
  std::string input_topic_;
  std::string output_topic_;
};

}  // namespace cvml_ros

#endif  // RYZEN_AI_CVML__DEPTH_ESTIMATION_NODE_HPP_
