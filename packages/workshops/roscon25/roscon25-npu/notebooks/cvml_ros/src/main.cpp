#include "rclcpp/rclcpp.hpp"
#include "cvml_ros/depth_estimation_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<cvml_ros::DepthEstimationNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception in main: %s", e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}