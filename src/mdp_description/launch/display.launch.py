import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ARGUMENTS = [
    DeclareLaunchArgument(
        'urdf_path',
        default_value=os.path.join(
            get_package_share_directory('mdp_description'),
            'urdf', 'mini_akm_robot.urdf'),
        description='Path to the robot URDF/xacro file'),
    DeclareLaunchArgument(
        'use_joint_state_gui', default_value='true',
        choices=['true', 'false'],
        description='Launch joint_state_publisher_gui instead of joint_state_publisher'),
    DeclareLaunchArgument(
        'use_rviz', default_value='true',
        choices=['true', 'false'],
        description='Launch RViz'),
    DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(
            get_package_share_directory('mdp_description'),
            'rviz', 'display.rviz'),
        description='Path to the RViz config file'),
]


def generate_launch_description():
    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('urdf_path')]),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(LaunchConfiguration('use_joint_state_gui')),
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('use_joint_state_gui')),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(joint_state_publisher_gui_node)
    ld.add_action(rviz_node)
    return ld
