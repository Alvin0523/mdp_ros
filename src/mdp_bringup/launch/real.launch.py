import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_description = get_package_share_directory('mdp_description')
    pkg_bringup = get_package_share_directory('mdp_bringup')

    urdf_file = os.path.join(pkg_description, 'urdf', 'mini_akm_real_robot.urdf')
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_desc},
            os.path.join(pkg_bringup, 'config', 'real_controller.yaml')
        ],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    ackermann_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ackermann_steering_controller', '--controller-manager', '/controller_manager'],
        remappings=[
            ('/ackermann_steering_controller/reference', '/cmd_vel')
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        ackermann_controller_spawner
    ])
