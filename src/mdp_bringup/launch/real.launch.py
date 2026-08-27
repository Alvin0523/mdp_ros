import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_description = get_package_share_directory('mdp_description')
    pkg_bringup = get_package_share_directory('mdp_bringup')

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial device for the STM32 USART3 bridge (varies by '
                     'host/driver - e.g. /dev/ttyACM0 on some machines).'
    )
    serial_port = LaunchConfiguration('serial_port')

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
        remappings=[
            ('/ackermann_steering_controller/reference', '/cmd_vel')
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
        output='screen'
    )

    serial_bridge = Node(
        package='mdp_hardware_bridge',
        executable='serial_bridge_node',
        parameters=[{'serial_port': serial_port, 'baud_rate': 115200}],
        output='screen'
    )

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        output='screen'
    )

    yolo_detector = Node(
        package='mdp_vision',
        executable='yolo_arrow_detector.py',
        parameters=[{'camera_topic': '/camera/image_raw'}],
        output='screen'
    )

    return LaunchDescription([
        serial_port_arg,
        robot_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        ackermann_controller_spawner,
        serial_bridge,
        camera_node,
        yolo_detector
    ])
