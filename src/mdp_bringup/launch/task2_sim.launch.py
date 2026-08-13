import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_description = get_package_share_directory('mdp_description')
    pkg_bringup = get_package_share_directory('mdp_bringup')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    urdf_file = os.path.join(pkg_description, 'urdf', 'mini_akm_robot.urdf')
    world_file = os.path.join(pkg_description, 'worlds', 'task2_arena.sdf')

    config_file = os.path.join(pkg_bringup, 'config', 'ackermann_controller.yaml')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read().replace(
            'package://mdp_bringup/config/ackermann_controller.yaml',
            config_file
        )

    # Environment variables for Gazebo meshes & system plugins
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_description, '..')
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r -v 4 {world_file}'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', robot_desc, '-name', 'mini_akm_robot', '-x', '0.0', '-y', '0.0', '-z', '0.05'],
        output='screen'
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
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

    yolo_detector = Node(
        package='mdp_control',
        executable='yolo_arrow_detector.py',
        parameters=[{'camera_topic': '/camera/image_raw'}],
        output='screen'
    )

    task2_runner = Node(
        package='mdp_control',
        executable='task2_runner.py',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        gz_resource_path,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        joint_state_broadcaster_spawner,
        ackermann_controller_spawner,
        yolo_detector,
        task2_runner
    ])
