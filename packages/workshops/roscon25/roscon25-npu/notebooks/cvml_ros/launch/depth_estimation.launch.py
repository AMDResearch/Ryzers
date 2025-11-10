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
        default_value='/depth_estimation/depth',
        description='Output depth topic'
    )

    # Create the depth estimation node
    depth_estimation_node = Node(
        package='cvml_ros',
        executable='depth_estimation_node',
        name='depth_estimation_node',
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
        }],
        output='screen'
    )

    return LaunchDescription([
        input_topic_arg,
        output_topic_arg,
        depth_estimation_node
    ])