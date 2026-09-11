import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Which YOLO model to load. Bare dir name under mdp_yolo/models/ (or an
    # absolute path). Defaults to the latest MDP-trained model, NOT stock COCO.
    #   best_ncnn_model_v2 -> MDP symbols, latest (default)
    #   best_ncnn_model_v1 -> MDP symbols, older
    #   yolo26n_ncnn_model -> stock YOLO26n COCO (person/car/...) - debug only
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='best_ncnn_model_v2',
        description='YOLO model dir name under mdp_yolo/models/ or an absolute path'
    )

    camera_node = Node(
        package='mdp_yolo',
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
        package='mdp_yolo',
        executable='yolo_detector.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'camera_topic': '/image_raw',
            'result_topic': '/yolo_result',
            'model_path': LaunchConfiguration('model')
        }]
    )

    return LaunchDescription([
        model_arg,
        camera_node,
        yolo_node
    ])
