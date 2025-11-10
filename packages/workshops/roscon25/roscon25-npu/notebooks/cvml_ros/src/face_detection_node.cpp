#include "cvml_ros/face_detection_node.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>

namespace cvml_ros
{

FaceDetectionNode::FaceDetectionNode(const rclcpp::NodeOptions & options)
: Node("face_detection_node", options)
{
  // Declare parameters
  this->declare_parameter("input_topic", "/camera/image_raw");
  this->declare_parameter("output_topic", "/face_detection/output");
  this->declare_parameter("fd_model_type", "fast");

  input_topic_ = this->get_parameter("input_topic").as_string();
  output_topic_ = this->get_parameter("output_topic").as_string();
  fd_model_type_ = this->get_parameter("fd_model_type").as_string();

  RCLCPP_INFO(this->get_logger(), "Input topic: %s", input_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Face detection model: %s", fd_model_type_.c_str());

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

    // Determine face detection model type
    amd::cvml::FaceDetector::FDModelType model_type =
      (fd_model_type_ == "precise") ?
        amd::cvml::FaceDetector::FDModelType::Precise :
        amd::cvml::FaceDetector::FDModelType::Fast;

    // Create face detector
    face_detector_ = std::make_unique<amd::cvml::FaceDetector>(context_, model_type);

    RCLCPP_INFO(this->get_logger(), "Ryzen AI Face Detection initialized");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize Ryzen AI: %s", e.what());
    throw;
  }

  // Create publisher
  output_pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);
  RCLCPP_INFO(this->get_logger(), "Created publisher on: %s", output_topic_.c_str());

  // Create subscriber
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    input_topic_, 10,
    std::bind(&FaceDetectionNode::imageCallback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "Created subscriber on: %s", input_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Face detection node started successfully");
}

FaceDetectionNode::~FaceDetectionNode()
{
  if (context_) {
    context_->Release();
  }
}

void FaceDetectionNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    // Convert ROS2 image to OpenCV
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat input_image = cv_ptr->image;
    cv::Mat output_image = input_image.clone();

    RCLCPP_INFO_ONCE(this->get_logger(), "Received first image: %dx%d", input_image.cols, input_image.rows);

    // Create CVML input image
    amd::cvml::Image amd_img(
      amd::cvml::Image::Format::kRGB,
      amd::cvml::Image::DataType::kUint8,
      input_image.cols,
      input_image.rows,
      input_image.data
    );

    // Detect faces
    auto faces = face_detector_->Detect(amd_img);

    // Draw detected faces
    cv::Scalar bbox_color(0, 255, 0);      // Green bounding box
    cv::Scalar landmark_color(0, 0, 255);  // Red landmarks
    cv::Scalar text_color(0, 255, 255);    // Yellow text
    int landmark_size = 2;

    for (size_t k = 0; k < faces.size(); ++k) {
      const amd::cvml::Face& curr_face = faces[k];

      // Draw bounding box
      cv::Rect cv_face(
        curr_face.face_.x_,
        curr_face.face_.y_,
        curr_face.face_.width_,
        curr_face.face_.height_
      );
      cv::rectangle(output_image, cv_face, bbox_color, 2);

      // Draw confidence score
      std::ostringstream out_text;
      out_text.precision(2);
      out_text << std::fixed << curr_face.confidence_score_;
      cv::putText(
        output_image,
        out_text.str(),
        cv::Point(cv_face.x, cv_face.y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        2
      );

      // Draw landmarks
      for (size_t j = 0; j < curr_face.landmarks_.size(); ++j) {
        const amd::cvml::Point2i& landmark = curr_face.landmarks_[j];
        cv::Point center(landmark.x_, landmark.y_);
        cv::circle(output_image, center, landmark_size, landmark_color, cv::FILLED);
      }
    }

    RCLCPP_INFO_ONCE(this->get_logger(), "Processed first frame with %zu faces", faces.size());

    // Convert to ROS2 message
    std_msgs::msg::Header header;
    header.stamp = msg->header.stamp;
    header.frame_id = msg->header.frame_id;

    sensor_msgs::msg::Image::SharedPtr output_msg = cv_bridge::CvImage(
      header,
      sensor_msgs::image_encodings::RGB8,
      output_image
    ).toImageMsg();

    // Publish output
    output_pub_->publish(*output_msg);
    RCLCPP_INFO_ONCE(this->get_logger(), "Published first output image");

  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error processing image: %s", e.what());
  }
}

}  // namespace cvml_ros
