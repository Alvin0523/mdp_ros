import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_mdp_description = get_package_share_directory('mdp_description')
    pkg_mdp_bringup = get_package_share_directory('mdp_bringup')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Ensure Gazebo Sim resolves package://mdp_description
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_mdp_description, '..')
    )

    # Ensure Gazebo Sim finds gz_ros2_control-system shared library
    pixi_lib_dir = os.path.abspath(os.path.join(pkg_ros_gz_sim, '../..', 'lib'))
    gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=pixi_lib_dir + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    )
    ign_plugin_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
        value=pixi_lib_dir + ':' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')
    )

    urdf_file = os.path.join(pkg_mdp_description, 'urdf', 'mini_akm_robot.urdf')
    controller_config = os.path.join(pkg_mdp_bringup, 'config', 'ackermann_controller.yaml')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # Substitute controller config path for Gazebo plugin
    processed_urdf = robot_desc.replace('package://mdp_bringup/config/ackermann_controller.yaml', controller_config)
    processed_urdf_path = '/tmp/mini_akm_robot_gazebo.urdf'
    with open(processed_urdf_path, 'w') as outfp:
        outfp.write(processed_urdf)

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': processed_urdf,
            'use_sim_time': True
        }]
    )

    # Gazebo Sim launch
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # Spawn robot in Gazebo
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', processed_urdf_path,
            '-name', 'mini_akm_robot',
            '-z', '0.1'
        ]
    )

    # Controller spawners
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

    # ROS <-> Gazebo Bridge
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gz_resource_path,
        gz_plugin_path,
        ign_plugin_path,
        robot_state_publisher,
        gz_sim,
        gz_spawn_entity,
        joint_state_broadcaster_spawner,
        ackermann_controller_spawner,
        ros_gz_bridge
    ])
