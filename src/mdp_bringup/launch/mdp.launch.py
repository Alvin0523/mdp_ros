"""The one bringup for the robot - real hardware or Gazebo, any task.

    ros2 launch mdp_bringup mdp.launch.py sim:=true  task:=1
    ros2 launch mdp_bringup mdp.launch.py sim:=false task:=2 vision:=false

(`pixi run sim ...` / `pixi run real ...` wrap these; any argument below can be
appended.)

Arguments
  sim        true  -> Gazebo robot + arena, everything on Gazebo's /clock
             false -> STM32 over `serial_port`, Pi camera            (default)
  task       0 bare car (manual drive, no runner), 1 explore + recognise,
             2 slalom                                                 (default 0)
  vision     true/false - camera + YOLO                               (default false)
  obstacles  tablet -> the real tablet over Bluetooth on `bluetooth_device`
             yaml   -> fake_tablet.py sends `layout` through the same bridge
             (default: yaml in sim, tablet on the robot)
  layout     obstacle layout YAML in tablet cells (config/test_obstacles.yaml).
             In sim it also places the Gazebo obstacles, so the planner and the
             arena always agree.
  start_x/start_y/start_yaw  arena-frame start pose (metres, rad).
             Default per task: 1/0 -> (0.15, 0.15, pi/2), 2 -> (0, 0, 0).
  gui        sim only - Gazebo window                                 (default true)
  model      YOLO model dir under mdp_vision/models/                  (default best_ncnn_model_v2)
  serial_port, bluetooth_device  real device paths

SHARED by sim and real, so they cannot drift apart: the Bluetooth bridge and
its log, robot_pose_feedback (ROBOT lines + /reset_pose), manual drive, the
task runners, YOLO, and the `map -> odom` transform. Only the robot underneath
differs:
  sim   Gazebo + gz_ros2_control; the controller owns odom -> base_footprint
        and the EKF (ekf_sim.yaml) only publishes /odometry/filtered.
  real  STM32 serial bridge + ros2_control_node; the EKF (ekf.yaml) owns
        odom -> base_footprint.

Every argument is resolved while the description is BUILT (not as a
LaunchConfiguration), so the graph contains only the nodes this run uses. The
start pose in particular must be one number for all its consumers (spawn,
map -> odom, the runner); `ros2 launch --show-args` still lists everything.
"""

import math
import os
import sys
import tempfile

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# obstacle_layout lives with the scripts: the source tree when this file is a
# (symlinked) source file, else the installed lib/mdp_bringup.
for _path in (os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'scripts'),
              os.path.join(get_package_prefix('mdp_bringup'), 'lib', 'mdp_bringup')):
    if os.path.isfile(os.path.join(_path, 'obstacle_layout.py')):
        sys.path.insert(0, _path)
        break
import obstacle_layout  # noqa: E402

FAKE_TABLET_LINK = '/tmp/mdp_fake_tablet'
START_POSE = {  # task -> default arena start pose
    '0': (0.15, 0.15, math.pi / 2.0),
    '1': (0.15, 0.15, math.pi / 2.0),   # rear axle in the start box, facing North
    '2': (0.0, 0.0, 0.0),               # task2_runner's arena conversion assumes this
}


def _launch_arg(name, default, argv=None):
    """Value of a `name:=value` command-line launch argument, or `default`.

    Read at build time on purpose - see this file's docstring."""
    prefix = f'{name}:='
    for entry in reversed(argv if argv is not None else sys.argv):
        if entry.startswith(prefix):
            return entry[len(prefix):]
    return default


def _true(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def generate_launch_description(argv=None):
    pkg_description = get_package_share_directory('mdp_description')
    pkg_bringup = get_package_share_directory('mdp_bringup')

    def arg(name, default):
        return _launch_arg(name, default, argv)

    sim = _true(arg('sim', 'false'))
    task = arg('task', '0')
    if task not in START_POSE:
        raise ValueError(f"task:={task} - expected 0, 1 or 2")
    vision = _true(arg('vision', 'false'))
    obstacles = arg('obstacles', 'yaml' if sim else 'tablet')
    if obstacles not in ('yaml', 'tablet'):
        raise ValueError(f"obstacles:={obstacles} - expected yaml or tablet")
    layout = arg('layout', os.path.join(pkg_bringup, 'config', 'test_obstacles.yaml'))
    gui = _true(arg('gui', 'true'))
    model = arg('model', 'best_ncnn_model_v2')
    serial_port = arg('serial_port', '/dev/ttyUSB0')
    bluetooth_device = arg('bluetooth_device', '/dev/rfcomm0')
    dx, dy, dyaw = START_POSE[task]
    start_x = float(arg('start_x', str(dx)))
    start_y = float(arg('start_y', str(dy)))
    start_yaw = float(arg('start_yaw', str(dyaw)))

    declared = [
        DeclareLaunchArgument('sim', default_value='false', description='true: Gazebo, false: real robot'),
        DeclareLaunchArgument('task', default_value='0', description='0 bare car (manual drive), 1 explore+recognise, 2 slalom'),
        DeclareLaunchArgument('vision', default_value='false', description='camera + YOLO'),
        DeclareLaunchArgument('obstacles', default_value='yaml in sim, tablet on real',
                              description='tablet: real tablet over Bluetooth; yaml: fake_tablet.py sends `layout`'),
        DeclareLaunchArgument('layout', default_value='config/test_obstacles.yaml',
                              description='obstacle layout YAML in tablet cells (also the sim arena obstacles)'),
        DeclareLaunchArgument('start_x', default_value='per task', description='arena start x, metres'),
        DeclareLaunchArgument('start_y', default_value='per task', description='arena start y, metres'),
        DeclareLaunchArgument('start_yaw', default_value='per task', description='arena start yaw, rad'),
        DeclareLaunchArgument('gui', default_value='true', description='sim: Gazebo window'),
        DeclareLaunchArgument('model', default_value='best_ncnn_model_v2', description='YOLO model under mdp_vision/models/'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0', description='real: STM32 USART3 device'),
        DeclareLaunchArgument('bluetooth_device', default_value='/dev/rfcomm0', description='tablet RFCOMM device'),
    ]
    sim_time = {'use_sim_time': sim}
    actions = []

    # ------------------------------------------------------------ robot ----
    if sim:
        pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
        controller_config = os.path.join(pkg_bringup, 'config', 'ackermann_controller.yaml')
        # Task 1's camera is mounted facing the car's left; otherwise forward
        # (see mini_akm_robot.urdf's camera_joint).
        camera_yaw = math.pi / 2.0 if task == '1' else 0.0
        with open(os.path.join(pkg_description, 'urdf', 'mini_akm_robot.urdf')) as f:
            robot_desc = f.read().replace(
                'package://mdp_bringup/config/ackermann_controller.yaml', controller_config
            ).replace('CAMERA_YAW_RAD', repr(camera_yaw))

        if task == '2':
            world_file = os.path.join(pkg_description, 'worlds', 'task2_arena.sdf')
        else:
            # Obstacles baked in from the layout - the same file the fake
            # tablet sends, see obstacle_layout.py.
            with open(os.path.join(pkg_description, 'worlds', 'task1_arena.sdf')) as f:
                world = obstacle_layout.world_sdf(f.read(), obstacle_layout.load(layout))
            world_file = os.path.join(tempfile.gettempdir(), 'mdp_task1_arena.sdf')
            with open(world_file, 'w') as f:
                f.write(world)

        pixi_lib_dir = os.path.abspath(os.path.join(pkg_ros_gz_sim, '../..', 'lib'))
        gz_launch = os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        actions += [
            SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg_description, '..')),
            SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
                                   pixi_lib_dir + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')),
            SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
                                   pixi_lib_dir + ':' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')),
            # Server and GUI as separate processes: on macOS `gz sim` cannot
            # run both in one (gazebosim/gz-sim#44); identical on Linux.
            IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
                                     launch_arguments={'gz_args': f'-s -r -v 4 {world_file}'}.items()),
        ]
        if gui:
            actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
                                                    launch_arguments={'gz_args': '-g'}.items()))
        actions += [
            Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
                 parameters=[{'robot_description': robot_desc}, sim_time]),
            Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-string', robot_desc, '-name', 'mini_akm_robot',
                            '-x', str(start_x), '-y', str(start_y), '-z', '0.05', '-Y', str(start_yaw)]),
            Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                 parameters=[sim_time],
                 arguments=[
                     '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                     # /cmd_vel is NOT bridged: the runner's TwistStamped goes
                     # straight to the controller inside gz_ros2_control, and a
                     # bridged gz.msgs.Twist would be a second, unstamped
                     # advertiser on /cmd_vel.
                     '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                     '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                     '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                     '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
                 ]),
            # gz-sim stamps sensor messages with its own scoped frame names and
            # there is no SDF override, so transforms from the URDF links the
            # sensors are mounted on hang them into the tree. The camera's is
            # NOT identity: /camera/image_raw and camera_info are in the optical
            # convention (z forward, x right, y down), while camera_link is a
            # body frame looking along +x with z up - roll -90, yaw -90 maps one
            # onto the other. As identity, Foxglove drew the view cone along
            # camera_link +z, i.e. at the sky.
            Node(package='tf2_ros', executable='static_transform_publisher', name='camera_sensor_frame_tf',
                 output='screen', parameters=[sim_time],
                 arguments=['--roll', str(-math.pi / 2.0), '--pitch', '0.0', '--yaw', str(-math.pi / 2.0),
                            '--frame-id', 'camera_link',
                            '--child-frame-id', 'mini_akm_robot/base_footprint/camera']),
            Node(package='tf2_ros', executable='static_transform_publisher', name='imu_sensor_frame_tf',
                 output='screen', parameters=[sim_time],
                 arguments=['--frame-id', 'base_link',
                            '--child-frame-id', 'mini_akm_robot/base_footprint/imu_sensor']),
            Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node', output='screen',
                 parameters=[os.path.join(pkg_bringup, 'config', 'ekf_sim.yaml')]),
            # One spawner for both controllers - two racing spawners gave
            # intermittent "No controllers loaded".
            Node(package='controller_manager', executable='spawner', output='screen',
                 arguments=['joint_state_broadcaster', 'ackermann_steering_controller',
                            '--controller-manager', '/controller_manager',
                            '--controller-manager-timeout', '60',
                            '--controller-ros-args', '-r /ackermann_steering_controller/reference:=/cmd_vel']),
        ]
        camera_topic = '/camera/image_raw'
    else:
        with open(os.path.join(pkg_description, 'urdf', 'mini_akm_real_robot.urdf')) as f:
            robot_desc = f.read()
        actions += [
            Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
                 parameters=[{'robot_description': robot_desc}]),
            Node(package='controller_manager', executable='ros2_control_node', output='screen',
                 parameters=[{'robot_description': robot_desc},
                             os.path.join(pkg_bringup, 'config', 'real_controller.yaml')],
                 remappings=[('/ackermann_steering_controller/reference', '/cmd_vel')]),
            Node(package='controller_manager', executable='spawner', output='screen',
                 arguments=['joint_state_broadcaster', 'ackermann_steering_controller',
                            '--controller-manager', '/controller_manager',
                            '--controller-manager-timeout', '30']),
            # The bridge's JointState is TopicBasedSystem's raw feedback input;
            # joint_state_broadcaster owns /joint_states.
            Node(package='mdp_bridge', executable='serial_bridge_node', output='screen',
                 parameters=[{'serial_port': serial_port, 'baud_rate': 115200}],
                 remappings=[('/joint_states', '/joint_states_raw')]),
            Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node', output='screen',
                 parameters=[os.path.join(pkg_bringup, 'config', 'ekf.yaml')]),
        ]
        if vision:
            actions.append(Node(
                package='mdp_vision', executable='rpi_cam_publisher.py', name='rpi_cam_publisher', output='screen',
                parameters=[{'image_width': 640, 'image_height': 480, 'frame_rate': 30.0,
                             'camera_topic': '/image_raw'}]))
        camera_topic = '/image_raw'

    # ----------------------------------------------------------- shared ----
    # map -> odom: `odom` is created at the start pose with identity
    # orientation, so this STATIC transform is exactly the start pose. The
    # same numbers go to the runner below.
    actions.append(Node(
        package='tf2_ros', executable='static_transform_publisher', name='map_to_odom_static_tf',
        output='screen', parameters=[sim_time],
        arguments=['--x', str(start_x), '--y', str(start_y), '--z', '0.0',
                   '--yaw', str(start_yaw), '--pitch', '0.0', '--roll', '0.0',
                   '--frame-id', 'map', '--child-frame-id', 'odom']))

    if vision:
        actions.append(Node(
            package='mdp_vision', executable='yolo_detector.py', output='screen',
            parameters=[{'camera_topic': camera_topic, 'model_path': model}, sim_time]))

    # Tablet link. With obstacles:=yaml the fake tablet provides the device
    # and plays the tablet's setup; the bridge cannot tell the difference.
    use_fake_tablet = obstacles == 'yaml' and task == '1'
    actions.append(Node(
        package='mdp_bridge', executable='bluetooth_bridge_node', output='screen',
        parameters=[{'device': FAKE_TABLET_LINK if use_fake_tablet else bluetooth_device}, sim_time]))
    if use_fake_tablet:
        actions.append(ExecuteProcess(
            cmd=[os.path.join(get_package_prefix('mdp_bringup'), 'lib', 'mdp_bringup', 'fake_tablet.py'),
                 layout, FAKE_TABLET_LINK],
            name='fake_tablet', output='screen'))

    # Base nodes, any task: ROBOT,<cell>,<cell>,<dir> to the tablet + the
    # /reset_pose service, and /bt_log for watching the tablet link.
    actions += [
        Node(package='mdp_bringup', executable='robot_pose_feedback.py', output='screen',
             parameters=[sim_time]),
        Node(package='mdp_bringup', executable='bt_monitor.py', name='bt_log', output='screen',
             arguments=['--quiet'], parameters=[sim_time]),
    ]

    if task == '0':
        # Bare car: the tablet's manual drive buttons, no runner.
        actions.append(Node(package='mdp_bringup', executable='manual_drive.py', output='screen',
                            parameters=[sim_time]))
    elif task == '1':
        # Idle in WAITING_FOR_SETUP until the tablet's DONE, plans, then holds
        # for `go` (tablet BEGIN / pixi run go).
        actions.append(Node(
            package='mdp_bringup', executable='task1_runner.py', output='screen',
            parameters=[os.path.join(pkg_bringup, 'config', 'occupancy_grid_viz.yaml'),
                        {'start_x': start_x, 'start_y': start_y, 'start_yaw': start_yaw}, sim_time]))
    elif task == '2':
        actions.append(Node(
            package='mdp_bringup', executable='task2_runner.py', output='screen',
            parameters=[sim_time]))

    return LaunchDescription(declared + actions)
