"""The one bringup for the robot - real hardware or Gazebo, any task.

    ros2 launch mdp_bringup mdp.launch.py sim:=true  task:=1
    ros2 launch mdp_bringup mdp.launch.py sim:=false task:=2 vision:=false   (no camera)

(`pixi run sim ...` / `pixi run real ...` wrap these; any argument below can be
appended.)

Arguments
  sim        true  -> Gazebo robot + arena, everything on Gazebo's /clock
             false -> STM32 over `serial_port`, Pi camera            (default)
  task       0 bare car (manual drive, no runner), 1 explore + recognise,
             2 slalom                                                 (default 0)
  vision     true/false - camera + YOLO                               (default true) 
  obstacles  tablet -> only the tablet (over Bluetooth on `bluetooth_device`)
             yaml   -> also publish `layout` once at startup, like `pixi run setup`
             (default: yaml in sim, tablet on the robot). The tablet link is up
             either way, so the tablet can always send a new set.
             Task 1 only.
  layout     obstacle layouts YAML (config/tasks.yaml: task1 in tablet cells,
             task2 in metres). In sim it also places the Gazebo obstacles, so
             the planner and the arena always agree.
  start_cell tablet cell COL,ROW (0..19) under base_link - the centre of the
             REAR AXLE - at start                         (default 1,1)
  start_dir  N / E / S / W, the way the car faces at start   (default N)
             Task 2 without start_cell keeps its arena origin, facing E.
  gui        sim only - Gazebo window                                 (default true)
  model      YOLO model dir under mdp_vision/models/                  (default mdp_v2_ncnn_model)
  serial_port, bluetooth_device  device paths    (default: config/bridges.yaml)
  log        quiet -> this terminal shows warnings/errors from everything, plus
                      the task runners and the tablet/STM32 bridges; Gazebo's
                      output goes to the log file only           (default)
             full   -> every node's full output, as before
             Either way every node's log is saved in ~/.ros/log/.

CONFIG (all in config/): navigation.yaml (everything that drives the car, all
the speeds), controller.yaml, ekf.yaml, bridges.yaml,
vision.yaml, tasks.yaml. The car's measured size and steering are the URDF's
(mdp_description/urdf/mdp_robot.urdf.xacro); this file copies them into the
controller's and the planner's parameters, so they exist once.

SHARED by sim and real, so they cannot drift apart: the controller and EKF
config (the EKF owns odom -> base_footprint in both), the Bluetooth bridge and its log, robot_pose_feedback (ROBOT lines +
/reset_pose), manual drive, the task runners, YOLO, and the `map -> odom`
transform. Only the robot underneath differs:
  sim   Gazebo + gz_ros2_control
  real  STM32 serial bridge + ros2_control_node
The robot itself is one description, mdp_description/urdf/mdp_robot.urdf.xacro
(sim:=true adds the Gazebo-only parts).

Every argument is resolved while the description is BUILT (not as a
LaunchConfiguration), so the graph contains only the nodes this run uses. The
start pose in particular must be one number for all its consumers (spawn,
map -> odom, the runner); `ros2 launch --show-args` still lists everything.
"""

import math
import os
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node as RosNode
import xacro
import yaml

from mdp_algorithm.utils.params import car_from_urdf
from mdp_bringup.utils import obstacle_layout

TASKS = ('0', '1', '2')
CELL_M = 0.10                  # tablet grid: 20 x 20 cells of 10 cm, (0,0) bottom-left
DIR_YAW = {'N': math.pi / 2.0, 'E': 0.0, 'S': -math.pi / 2.0, 'W': math.pi}
DEFAULT_START_CELL = '1,1'     # base_link (rear axle centre) over cell (1,1): inside the 4x4-cell start box
DEFAULT_START_DIR = 'N'
TASK2_ORIGIN_POSE = (0.0, 0.0, 0.0)   # task2_runner's arena conversion assumes this


def start_pose(cell: str, direction: str):
    """'col,row' + N/E/S/W -> arena (x, y, yaw) of base_link, metres/rad: the
    centre of that cell. The only place cells become metres for the start."""
    try:
        col, row = (int(v) for v in cell.replace(' ', '').split(','))
    except ValueError:
        raise ValueError(f"start_cell:={cell} - expected COL,ROW, e.g. 1,1")
    if not (0 <= col <= 19 and 0 <= row <= 19):
        raise ValueError(f"start_cell:={cell} - cells are 0..19")
    d = direction.strip().upper()
    if d not in DIR_YAW:
        raise ValueError(f"start_dir:={direction} - expected N, E, S or W")
    return ((col + 0.5) * CELL_M, (row + 0.5) * CELL_M, DIR_YAW[d])


def _launch_arg(name, default, argv=None):
    """Value of a `name:=value` command-line launch argument, or `default`.

    Read at build time on purpose - see this file's docstring."""
    prefix = f'{name}:='
    for entry in reversed(argv if argv is not None else sys.argv):
        if entry.startswith(prefix):
            return entry[len(prefix):]
    return default


# Nodes whose normal (INFO) output is the story of the run; with log:=quiet
# every other node shows only its warnings and errors in the terminal.
_STORY_NODES = {'task1_runner', 'task2_runner', 'bluetooth_bridge_node',
                'serial_bridge_node', 'manual_drive', 'sim_obstacles'}


# Nodes whose INFO lines go to /rosout only (not the terminal), at any log:=.
_ROSOUT_ONLY_NODES = {'bt_monitor'}


def _temp_file(name: str, text: str) -> str:
    """Write a generated file for this launch (per-launch name, so two bringups
    on one machine never share it)."""
    path = os.path.join(tempfile.gettempdir(), f'mdp_{name}_{os.getpid()}')
    with open(path, 'w') as f:
        f.write(text)
    return path


def _controller_config(template: str, car: dict) -> str:
    """controller.yaml with the car's dimensions (from the URDF) filled in."""
    with open(template) as f:
        config = yaml.safe_load(f)
    config['ackermann_steering_controller']['ros__parameters'].update({
        'wheelbase': car['wheelbase'],
        'traction_track_width': car['rear_track'],
        'steering_wheels_radius': car['wheel_radius'],
        'traction_wheels_radius': car['wheel_radius'],
    })
    return _temp_file('controller.yaml', yaml.safe_dump(config))


def _true(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def generate_launch_description(argv=None):
    pkg_description = get_package_share_directory('mdp_description')
    pkg_bringup = get_package_share_directory('mdp_bringup')

    def arg(name, default):
        return _launch_arg(name, default, argv)

    sim = _true(arg('sim', 'false'))
    task = arg('task', '0')
    if task not in TASKS:
        raise ValueError(f"task:={task} - expected 0, 1 or 2")
    vision = _true(arg('vision', 'true'))
    obstacles = arg('obstacles', 'yaml' if sim else 'tablet')
    if obstacles not in ('yaml', 'tablet'):
        raise ValueError(f"obstacles:={obstacles} - expected yaml or tablet")
    layout = arg('layout', os.path.join(pkg_bringup, 'config', 'tasks.yaml'))
    gui = _true(arg('gui', 'true'))
    model = arg('model', 'mdp_v2_ncnn_model')
    quiet = arg('log', 'quiet') != 'full'
    serial_port = arg('serial_port', '')              # '' = config/bridges.yaml
    bluetooth_device = arg('bluetooth_device', '')
    cell, direction = arg('start_cell', ''), arg('start_dir', DEFAULT_START_DIR)
    if task == '2' and not cell:
        start_x, start_y, start_yaw = TASK2_ORIGIN_POSE
    else:
        start_x, start_y, start_yaw = start_pose(cell or DEFAULT_START_CELL, direction)

    declared = [
        DeclareLaunchArgument('sim', default_value='false', description='true: Gazebo, false: real robot'),
        DeclareLaunchArgument('task', default_value='0', description='0 bare car (manual drive), 1 explore+recognise, 2 slalom'),
        DeclareLaunchArgument('vision', default_value='true', description='camera + YOLO (false: without)'),
        DeclareLaunchArgument('obstacles', default_value='yaml in sim, tablet on real',
                              description='tablet: only the tablet; yaml: also publish `layout` once at startup (task 1)'),
        DeclareLaunchArgument('layout', default_value='config/tasks.yaml',
                              description='obstacle layouts (task1 cells, task2 metres) - planner AND sim arena'),
        DeclareLaunchArgument('start_cell', default_value=DEFAULT_START_CELL,
                              description='COL,ROW tablet cell under the rear axle centre (base_link) at start'),
        DeclareLaunchArgument('start_dir', default_value=DEFAULT_START_DIR, description='N/E/S/W facing at start'),
        DeclareLaunchArgument('gui', default_value='true', description='sim: Gazebo window'),
        DeclareLaunchArgument('model', default_value='mdp_v2_ncnn_model', description='YOLO model under mdp_vision/models/'),
        DeclareLaunchArgument('serial_port', default_value='config/bridges.yaml', description='real: STM32 USART3 device'),
        DeclareLaunchArgument('bluetooth_device', default_value='config/bridges.yaml', description='tablet RFCOMM device'),
        DeclareLaunchArgument('log', default_value='quiet',
                              description='quiet: warnings/errors + runner/bridges only; full: everything'),
    ]

    def Node(**kw):
        """launch_ros Node, with this launch's log:= policy applied."""
        if quiet:
            kw['output_format'] = '{line}'   # no '[name-N] ' launch prefix
            if kw.get('executable') not in _STORY_NODES | _ROSOUT_ONLY_NODES:
                kw['ros_arguments'] = list(kw.get('ros_arguments', [])) + ['--log-level', 'warn']
        return RosNode(**kw)

    sim_time = {'use_sim_time': sim}
    actions = []
    if quiet:
        # One short line per message: '[INFO] [task1_runner]: Leg 1/4 planned'.
        actions.append(SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] [{name}]: {message}'))

    # ------------------------------------------------------------ robot ----
    config = os.path.join(pkg_bringup, 'config')
    xacro_file = os.path.join(pkg_description, 'urdf', 'mdp_robot.urdf.xacro')
    car = car_from_urdf(xacro_file)   # the car's measured numbers - the URDF is their one copy
    controller_config = _controller_config(os.path.join(config, 'controller.yaml'), car)
    bridges = os.path.join(config, 'bridges.yaml')
    navigation = os.path.join(config, 'navigation.yaml')
    # The robot, sim or real: one xacro. Task 1's camera looks out of the car's
    # left side, otherwise forward.
    robot_desc = xacro.process_file(
        xacro_file,
        mappings={'sim': str(sim).lower(), 'camera_yaw': repr(math.pi / 2.0 if task == '1' else 0.0),
                  'controller_config': controller_config}).toxml()

    if sim:
        pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

        # Obstacles baked in from the layout - the same file the runner plans
        # with (see obstacle_layout.py).
        world_name = 'task2_arena' if task == '2' else 'task1_arena'
        with open(os.path.join(pkg_description, 'worlds', f'{world_name}.sdf')) as f:
            world = obstacle_layout.world_sdf(
                f.read(), obstacle_layout.load(layout, 'task2' if task == '2' else 'task1'))
        world_file = _temp_file(f'{world_name}.sdf', world)

        pixi_lib_dir = os.path.abspath(os.path.join(pkg_ros_gz_sim, '../..', 'lib'))
        # Gazebo started directly (not via ros_gz_sim's gz_sim.launch.py, which
        # always prints to the screen): server and GUI as separate processes (on
        # macOS `gz sim` cannot run both in one, gazebosim/gz-sim#44). With
        # log:=quiet their output - including gz_ros2_control's INFO lines,
        # which cannot be given a log level - goes to the log file only; a
        # crash is still reported here by launch itself.
        gz_output = 'own_log' if quiet else 'screen'   # own_log: stdout AND stderr to a file (plain 'log' still shows stderr)
        actions += [
            SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg_description, '..')),
            SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
                                   pixi_lib_dir + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')),
            SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
                                   pixi_lib_dir + ':' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')),
            ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '-v', '2' if quiet else '4', world_file],
                           name='gazebo', output=gz_output),
        ]
        if gui:
            actions.append(ExecuteProcess(cmd=['gz', 'sim', '-g'], name='gazebo_gui', output=gz_output))
        actions += [
            Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
                 parameters=[{'robot_description': robot_desc}, sim_time]),
            Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-string', robot_desc, '-name', 'mini_akm_robot',
                            '-x', str(start_x), '-y', str(start_y), '-z', '0.05', '-Y', str(start_yaw)]),
            Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                 parameters=[sim_time],
                 # Gazebo's TRUE pose of every moving model (the car is
                 # 'mini_akm_robot', its base_footprint, in the world = arena
                 # frame) on /sim/ground_truth - to check odometry/EKF and
                 # arrival accuracy. Deliberately NOT /tf.
                 remappings=[(f'/world/{world_name}/dynamic_pose/info', '/sim/ground_truth')],
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
                     f'/world/{world_name}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
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
                 arguments=['--frame-id', 'imu_link',   # the IMU sensor sits on imu_link (URDF)
                            '--child-frame-id', 'mini_akm_robot/base_footprint/imu_sensor']),

            # One spawner for both controllers - two racing spawners gave
            # intermittent "No controllers loaded".
            Node(package='controller_manager', executable='spawner', output='screen',
                 arguments=['joint_state_broadcaster', 'ackermann_steering_controller',
                            '--controller-manager', '/controller_manager',
                            '--controller-manager-timeout', '60',
                            '--controller-ros-args',
                            '-r /ackermann_steering_controller/reference:=/cmd_vel']),
        ]
        camera_topic = '/camera/image_raw'
    else:
        actions += [
            Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
                 parameters=[{'robot_description': robot_desc}]),
            Node(package='controller_manager', executable='ros2_control_node', output='screen',
                 parameters=[{'robot_description': robot_desc}, controller_config],
                 remappings=[('/ackermann_steering_controller/reference', '/cmd_vel')]),
            Node(package='controller_manager', executable='spawner', output='screen',
                 arguments=['joint_state_broadcaster', 'ackermann_steering_controller',
                            '--controller-manager', '/controller_manager',
                            '--controller-manager-timeout', '30']),
            # The bridge's JointState is TopicBasedSystem's raw feedback input;
            # joint_state_broadcaster owns /joint_states.
            Node(package='mdp_bridge', executable='serial_bridge_node', output='screen',
                 parameters=[bridges] + ([{'serial_port': serial_port}] if serial_port else []),
                 remappings=[('/joint_states', '/joint_states_raw')]),
        ]
        camera_topic = '/image_raw'

    # ----------------------------------------------------------- shared ----
    actions.append(Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                        output='screen', parameters=[os.path.join(config, 'ekf.yaml'), sim_time]))
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
        # Camera (real only - Gazebo has its own) + YOLO: launch/vision.launch.py.
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_bringup, 'launch', 'vision.launch.py')),
            launch_arguments={'model': model, 'camera': str(not sim).lower(), 'camera_topic': camera_topic,
                              'use_sim_time': str(sim).lower(),
                              'log_level': 'warn' if quiet else 'info'}.items()))

    # Tablet link - the real tablet, in sim as on the robot.
    actions.append(Node(
        package='mdp_bridge', executable='bluetooth_bridge_node', output='screen',
        parameters=[bridges] + ([{'device': bluetooth_device}] if bluetooth_device else []) + [sim_time]))
    if sim and task == '1':
        # A tablet layout that differs from the file replaces Gazebo's blocks.
        actions.append(Node(package='mdp_bringup', executable='sim_obstacles', output='screen',
                            parameters=[{'layout': layout, 'world': 'task1_arena'}, sim_time]))
    if obstacles == 'yaml' and task == '1':
        # The layout, published once to /obstacle_setup after task1_runner
        # subscribes - exactly what `pixi run setup` does.
        actions.append(Node(
            package='mdp_bringup', executable='publish_obstacles', output='screen',
            arguments=[layout], parameters=[sim_time]))

    # Base nodes, any task: ROBOT,<cell>,<cell>,<dir> to the tablet + the
    # /reset_pose service, and bt_monitor - the tablet link's traffic as log
    # lines on /rosout (`pixi run btlog`), kept off this terminal.
    actions += [
        Node(package='mdp_bringup', executable='robot_pose_feedback', output='screen',
             parameters=[sim_time]),
        # One-glance health on /diagnostics (STM32/tablet links, sensor rates,
        # camera/YOLO, runner) - Foxglove "Diagnostics" panels.
        Node(package='mdp_bringup', executable='health_monitor', output='screen',
             parameters=[{'sim': sim, 'vision': vision, 'task': task, 'camera_topic': camera_topic},
                         sim_time]),
        Node(package='mdp_bringup', executable='bt_monitor', output='screen',
             ros_arguments=['--disable-stdout-logs'], parameters=[sim_time]),
    ]

    if task == '0':
        # Bare car: the tablet's manual drive buttons, no runner.
        actions.append(Node(package='mdp_bringup', executable='manual_drive', output='screen',
                            parameters=[navigation, sim_time]))
    elif task == '1':
        # Idle in WAITING_FOR_SETUP until the tablet's DONE, plans, then holds
        # for `go` (tablet BEGIN / pixi run go).
        actions.append(Node(
            package='mdp_bringup', executable='task1_runner', output='screen',
            parameters=[navigation,
                        {f'robot.{k}': car[k] for k in ('wheelbase', 'steering_limit_left', 'steering_limit_right')},
                        {'start_x': start_x, 'start_y': start_y, 'start_yaw': start_yaw}, sim_time]))
    elif task == '2':
        actions.append(Node(
            package='mdp_bringup', executable='task2_runner', output='screen',
            parameters=[navigation, {'layout': layout}, sim_time]))

    return LaunchDescription(declared + actions)
