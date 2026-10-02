"""Camera + YOLO - on its own (`pixi run vision`) or included by mdp.launch.py
(vision:=true). Settings: config/vision.yaml.

    ros2 launch mdp_bringup vision.launch.py                       Pi camera + YOLO
    ros2 launch mdp_bringup vision.launch.py model:=mdp_v1_ncnn_model
    ros2 launch mdp_bringup vision.launch.py camera:=false camera_topic:=/camera/image_raw
                                                                  YOLO on another camera (sim)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('mdp_bringup'), 'config', 'vision.yaml')
    arg = LaunchConfiguration
    common = {'use_sim_time': arg('use_sim_time')}
    ros_args = ['--log-level', arg('log_level')]
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='mdp_v2_ncnn_model',
                              description='YOLO model dir under mdp_vision/models/, or an absolute path'),
        DeclareLaunchArgument('camera', default_value='true', description='start the Pi camera'),
        DeclareLaunchArgument('yolo', default_value='true',
                              description='start YOLO (false: the laptop runs it - `pixi run car` / `base`)'),
        DeclareLaunchArgument('camera_topic', default_value='/image_raw', description='image topic YOLO reads'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        Node(package='mdp_vision', executable='rpi_cam_publisher', output='screen',
             condition=IfCondition(arg('camera')), parameters=[config, common], ros_arguments=ros_args),
        Node(package='mdp_vision', executable='yolo_detector', output='screen',
             condition=IfCondition(arg('yolo')),
             parameters=[config, {'model_path': arg('model'), 'camera_topic': arg('camera_topic')}, common],
             ros_arguments=ros_args),
    ])
