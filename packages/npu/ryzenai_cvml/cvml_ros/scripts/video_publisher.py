#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        
        # Declare parameters
        self.declare_parameter('video_path', '/path/to/video.mp4')
        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('loop', True)
        
        # Get parameters
        video_path = self.get_parameter('video_path').value
        topic = self.get_parameter('topic').value
        fps = self.get_parameter('fps').value
        self.loop = self.get_parameter('loop').value
        
        # Create publisher
        self.publisher_ = self.create_publisher(Image, topic, 10)
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open video: {video_path}')
            raise RuntimeError(f'Failed to open video: {video_path}')
        
        self.get_logger().info(f'Publishing video from: {video_path}')
        self.get_logger().info(f'Publishing to topic: {topic}')
        
        # Create timer to publish frames
        self.bridge = CvBridge()
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                self.get_logger().info('Video finished')
                rclpy.shutdown()
                return
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        
        # Publish
        self.publisher_.publish(msg)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
