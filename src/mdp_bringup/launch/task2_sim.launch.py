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

    # Camera stays forward-facing for Task 2 (arrow detection ahead) - matches
    # real hardware's default mount; Task 1 overrides this URDF placeholder to
    # face left instead, see mini_akm_robot.urdf's camera_joint comment.
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read().replace(
            'package://mdp_bringup/config/ackermann_controller.yaml',
            config_file
        ).replace('CAMERA_YAW_RAD', '0.0')

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

    # map -> odom, the arena frame's only link to the robot's dead-reckoned
    # pose - same reasoning as task1_sim.launch.py's map_to_odom. This task
    # spawns at the arena origin facing +X (no -Y given above, so yaw 0),
    # which is the value task2_runner.py's own arena_frame conversion now
    # depends on (see that file's odom_callback).
    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
            '--frame-id', 'map', '--child-frame-id', 'odom'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # gz-sim's IMU/camera sensor plugins stamp their own auto-generated scoped
    # frame_id into message headers with no TF parent - see the identical
    # comment in task1_sim.launch.py for why a static identity transform from
    # the real URDF link is the fix rather than an SDF-level override.
    camera_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_sensor_frame_tf',
        arguments=[
            '--frame-id', 'camera_link',
            '--child-frame-id', 'mini_akm_robot/base_footprint/camera'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    imu_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_sensor_frame_tf',
        arguments=[
            '--frame-id', 'base_link',
            '--child-frame-id', 'mini_akm_robot/base_footprint/imu_sensor'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Fuses wheel odometry + the simulated IMU into /odometry/filtered, which
    # task2_runner.py now consumes (see that file's odom_callback) instead of
    # a Gazebo-only ground-truth TF - same estimator role as task1_sim's EKF.
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[os.path.join(pkg_bringup, 'config', 'ekf_sim.yaml')],
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
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # Simulated IMU (mini_akm_robot.urdf's base_link sensor) -> same
            # topic name real hardware's mdp_bridge publishes, so ekf_sim.yaml
            # can fuse it exactly like ekf.yaml does on real hardware.
            '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # ONE spawner loads + activates BOTH controllers - matches
    # task1_sim.launch.py's fix. Relying on the gz_ros2_control plugin to
    # auto-load them itself proved unreliable here too: verified live,
    # controller_manager initializes and activates the GazeboSimSystem
    # hardware component, but never logs "Loading controller" for either
    # controller, so `ros2 control list_controllers` reports none loaded and
    # /cmd_vel never reaches the wheels. The /cmd_vel remap goes on the
    # ackermann controller node via --controller-ros-args.
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'ackermann_steering_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60',
            '--controller-ros-args',
            '-r /ackermann_steering_controller/reference:=/cmd_vel',
        ],
        output='screen'
    )

    yolo_detector = Node(
        package='mdp_yolo',
        executable='yolo_detector.py',
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'model_path': LaunchConfiguration('model')
        }],
        output='screen'
    )

    # Obstacles (with arrow decals) are baked directly into task2_arena.sdf -
    # same reasoning as task1_arena.sdf, see that file's header comment. No
    # runtime spawner needed.

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
        map_to_odom,
        camera_frame_tf,
        imu_frame_tf,
        ekf_node,
        gz_bridge,
        controller_spawner,
        yolo_detector,
        task2_runner
    ])
