import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Task 1 sim - full self-contained run (mirrors `pixi run real1` + the
    tablet's obstacle-setup, minus the `go`):

      pixi run sim1  -> Gazebo arena (obstacles + textured decals baked into
                        task1_arena.sdf) + robot + camera/vision + task1 runner
                        (idle) + planner-feed. The runner receives the obstacle
                        layout and plans, then HOLDS.
      pixi run go     -> calls /start_run -> the car drives.

    Obstacles live directly in task1_arena.sdf (edit that file to change the
    layout) - baked in at world-load time so their decal textures render
    (runtime-spawned decals show up black in ogre2). The planner is still fed
    the layout via publish_test_obstacles.py -> /obstacle_setup, the tablet
    stand-in. Everything runs use_sim_time:=true.

    Coordinate convention matches the planner: world origin = arena bottom-left
    corner; robot spawns at the (0.15, 0.15) start pose facing +Y (North). That
    pose is declared ONCE below (START_X/START_Y/START_YAW) and feeds all three
    of its consumers - the Gazebo spawn, the static `map -> odom` transform and
    task1_runner's planning start pose - so they cannot drift apart.
    """
    pkg_description = get_package_share_directory('mdp_description')
    pkg_bringup = get_package_share_directory('mdp_bringup')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # The robot's start pose in the ARENA frame (`map`), and the single source
    # for it in this launch file. Previously the same three numbers appeared as
    # the `create` spawn arguments here and again as task1_runner's own
    # `start_pose` literal; whenever they disagreed the arena rendered offset
    # from the robot. Consumers, all below: the Gazebo spawn (`-x -y -Y`), the
    # static `map -> odom` transform (T_map_odom IS this pose, since `odom` is
    # created at the spawn pose with identity orientation), and task1_runner's
    # `start_x`/`start_y`/`start_yaw` parameters.
    START_X, START_Y = 0.15, 0.15
    START_YAW = math.pi / 2.0        # facing arena +Y ("North"), as the planner assumes

    model_arg = DeclareLaunchArgument(
        'model',
        default_value='best_ncnn_model_v2',
        description='YOLO model dir name under mdp_yolo/models/ or an absolute path'
    )

    urdf_file = os.path.join(pkg_description, 'urdf', 'mini_akm_robot.urdf')
    # Obstacles are baked directly into this world file - loaded as-is.
    world_file = os.path.join(pkg_description, 'worlds', 'task1_arena.sdf')
    controller_config = os.path.join(pkg_bringup, 'config', 'ackermann_controller.yaml')
    obstacles_config = os.path.join(pkg_bringup, 'config', 'test_obstacles.yaml')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read().replace(
            'package://mdp_bringup/config/ackermann_controller.yaml',
            controller_config
        )

    # Gazebo resource path (resolves package://mdp_description meshes + the
    # symbol textures the spawner references) and the gz_ros2_control plugin.
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

    # Server + GUI as separate processes (see task2_sim.launch.py note).
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

    # Robot spawned at the declared start pose above. z lifted slightly so it
    # settles onto the floor.
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_desc,
            '-name', 'mini_akm_robot',
            '-x', str(START_X), '-y', str(START_Y), '-z', '0.05',
            '-Y', str(START_YAW)
        ],
        output='screen'
    )

    # map -> odom, the arena frame's only link to the robot's dead-reckoned pose.
    #
    # `odom` is created at the spawn pose with identity orientation, so the arena
    # pose of `odom` is exactly the start pose - hence a STATIC transform: with no
    # absolute localization source there is nothing to correct it with, and all
    # drift belongs in `odom -> base` where REP-105 puts it.
    #
    # Ownership is deliberate and exclusive: this node and nothing else. The
    # ackermann_steering_controller keeps sole ownership of odom -> base_footprint
    # (ackermann_controller.yaml, enable_odom_tf: true), robot_state_publisher
    # only publishes URDF joints, and the Gazebo /tf bridge carries no such edge.
    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        arguments=[
            '--x', str(START_X), '--y', str(START_Y), '--z', '0.0',
            '--yaw', str(START_YAW), '--pitch', '0.0', '--roll', '0.0',
            '--frame-id', 'map', '--child-frame-id', 'odom'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # The publisher of /odometry/filtered, which task1_runner subscribes to. The
    # sim previously ran no EKF at all, so that subscription never fired and the
    # runner's pose sat at its constructor default for the whole run - the arena
    # frame fix could not be exercised in sim because there was no pose to
    # transform.
    #
    # config/ekf_sim.yaml, NOT config/ekf.yaml: a separate overlay so hardware
    # localization is provably untouched, and because three values must differ in
    # sim - `publish_tf: false` (the ackermann_steering_controller keeps sole
    # ownership of odom -> base_footprint; a second broadcaster of one edge is the
    # exact mistake this arrangement is set up to avoid), `base_link_frame:
    # base_footprint` (the sim URDF's root link), and no `imu0` (nothing bridges
    # an IMU out of Gazebo). See that file's header for the reasoning.
    #
    # This node is a pure estimator here: it publishes a topic and broadcasts no
    # TF whatsoever.
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
            # NOTE: /cmd_vel is intentionally NOT bridged. The task runner
            # publishes /cmd_vel as geometry_msgs/TwistStamped, and the
            # ackermann_steering_controller (run inside sim by gz_ros2_control,
            # use_stamped_vel:=true) subscribes to it directly in ROS - the
            # command never needs to cross into Gazebo transport. Bridging it
            # as gz.msgs.Twist added a second, unstamped advertiser on
            # /cmd_vel -> Foxglove "schema does not match" + the runner's
            # TwistStamped never reaching the controller.
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # ONE spawner loads + activates BOTH controllers. Relying on the
    # gz_ros2_control plugin to auto-load them proved unreliable (intermittent
    # `ros2 control list_controllers` -> "No controllers loaded", so teleop /
    # cmd_vel does nothing). A single spawner (not two racing ones) is the
    # deterministic loader; --controller-manager-timeout waits for the
    # plugin-hosted manager to come up. The /cmd_vel remap goes on the
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

    # (Obstacles + decals are baked directly into task1_arena.sdf, so no
    # runtime spawning - the textures render because the entities exist at
    # world load. Edit that SDF to change the layout.)

    # Task 1 runner (the brain). Comes up idle (WAITING_FOR_SETUP), plans once
    # it receives the obstacle layout below, then HOLDS in WAITING_FOR_GO until
    # `pixi run go`. use_sim_time so it tracks Gazebo's /clock.
    task1_runner = Node(
        package='mdp_bringup',
        executable='task1_runner.py',
        parameters=[
            os.path.join(pkg_bringup, 'config', 'occupancy_grid_viz.yaml'),
            # Same start pose the robot is spawned at and the map -> odom
            # transform publishes, so the planner's first waypoint is where the
            # car actually is.
            {'start_x': START_X, 'start_y': START_Y, 'start_yaw': START_YAW},
            {'use_sim_time': True}
        ],
        output='screen'
    )

    # Planner-feed: the SAME test_obstacles.yaml the visual spawner uses,
    # published to /obstacle_setup (id/x/y/facing only). This is real's phase-2
    # `setup` done automatically at launch - the runner plans off it, then
    # holds for `go`. (The symbol images stay out of the planner, by design.)
    publish_obstacles = Node(
        package='mdp_bringup',
        executable='publish_test_obstacles.py',
        arguments=[obstacles_config],
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
        ekf_node,
        gz_bridge,
        controller_spawner,
        yolo_detector,
        task1_runner,
        publish_obstacles
    ])
