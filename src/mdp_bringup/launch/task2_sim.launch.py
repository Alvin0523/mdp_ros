import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Which YOLO model the sim detector loads. Defaults to the latest MDP-
    # trained model (best_ncnn_model_v2); override e.g.
    # `model:=best_ncnn_model_v1` or `model:=yolo26n_ncnn_model` (stock COCO).
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='best_ncnn_model_v2',
        description='YOLO model dir name under mdp_yolo/models/ or an absolute path'
    )

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

    pixi_lib_dir = os.path.abspath(os.path.join(pkg_ros_gz_sim, '../..', 'lib'))
    gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=pixi_lib_dir + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    )
    ign_plugin_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
        value=pixi_lib_dir + ':' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')
    )

    # Server and GUI are launched as separate processes: on macOS, `gz sim`
    # cannot run server + GUI together in one process (gazebosim/gz-sim#44).
    # Splitting them works identically on Linux, so we always do it.
    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-s -r -v 4 {world_file}'}.items()
    )

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g'}.items()
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
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # /cmd_vel NOT bridged - the runner publishes TwistStamped and the
            # ackermann controller (gz_ros2_control, use_stamped_vel) subscribes
            # to it in ROS directly. Bridging it as gz.msgs.Twist added a
            # mismatched second advertiser. See task1_sim.launch.py note.
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # NO controller spawner - the gz_ros2_control plugin in the URDF loads +
    # activates both controllers itself from ackermann_controller.yaml (see
    # the note in task1_sim.launch.py). A spawner is redundant and dies with
    # "already loaded / Failed to configure".

    yolo_detector = Node(
        package='mdp_yolo',
        executable='yolo_detector.py',
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'model_path': LaunchConfiguration('model')
        }],
        output='screen'
    )

    # Spawns the obstacles (with symbol-image decals) from test_obstacles.yaml
    # into the running Gazebo world - the camera/vision half (what YOLO sees).
    # It waits on Gazebo's spawn service, so ordering against gz_sim_server is
    # handled internally (no timing hack).
    obstacles_config = os.path.join(pkg_bringup, 'config', 'test_obstacles.yaml')
    spawn_obstacles = Node(
        package='mdp_bringup',
        executable='spawn_obstacles.py',
        arguments=[obstacles_config, '--world', 'task2_arena'],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Task 2 runner (the brain). Comes up idle (WAITING_FOR_START) - there is
    # NO obstacle setup for task 2 (fixed slalom), so it just holds until
    # `pixi run go` (the /start_run service). use_sim_time tracks Gazebo's
    # /clock. So sim2 = this bring-up, then `go`.
    task2_runner = Node(
        package='mdp_bringup',
        executable='task2_runner.py',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        model_arg,
        gz_resource_path,
        gz_plugin_path,
        ign_plugin_path,
        gz_sim_server,
        gz_sim_gui,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        yolo_detector,
        spawn_obstacles,
        task2_runner
    ])
