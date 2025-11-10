# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )

    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/face_detection/output',
        description='Output image topic with face detection overlay'
    )

    fd_model_type_arg = DeclareLaunchArgument(
        'fd_model_type',
        default_value='fast',
        description='Face detection model type: precise or fast'
    )

    # Create the face detection node
    face_detection_node = Node(
        package='cvml_ros',
        executable='face_detection_node',
        name='face_detection_node',
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'fd_model_type': LaunchConfiguration('fd_model_type'),
        }],
        output='screen'
    )

    return LaunchDescription([
        input_topic_arg,
        output_topic_arg,
        fd_model_type_arg,
        face_detection_node
    ])
