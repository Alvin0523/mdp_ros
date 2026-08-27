import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_node = Node(
        package='mdp_vision',
        executable='camera_publisher.py',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'video_device': 0,
            'image_width': 640,
            'image_height': 480,
            'frame_rate': 30.0,
            'camera_topic': '/image_raw'
        }]
    )

    yolo_node = Node(
        package='mdp_vision',
        executable='yolo_arrow_detector.py',
        name='yolo_arrow_detector',
        output='screen',
        parameters=[{
            'camera_topic': '/image_raw',
            'result_topic': '/yolo_result'
        }]
    )

    return LaunchDescription([
        camera_node,
        yolo_node
    ])
