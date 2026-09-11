import math
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node

# Where the car is placed in the arena, in the arena (`map`) frame: rear axle at
# the inner corner of the 40x40cm start box, facing arena +Y ("North"), which is
# the orientation the planner assumes. This is the DEFAULT only - the car may be
# placed anywhere inside that box, so override per run to match the actual
# placement:
#
#   ros2 launch mdp_bringup real.launch.py start_x:=0.2 start_y:=0.1 start_yaw:=1.5708
#
# The value feeds the static `map -> odom` transform and task1_runner's planning
# start pose from one place; if they disagree, the arena renders offset from the
# robot, which is precisely the bug this arrangement exists to prevent.
DEFAULT_START_X = 0.15
DEFAULT_START_Y = 0.15
DEFAULT_START_YAW = math.pi / 2.0


def _launch_arg(name, default, argv=None):
    """Value of a `name:=value` command-line launch argument, or `default`.

    Returns a plain string. The start pose is needed while this description is
    being BUILT, not only when it is evaluated: it goes onto
    static_transform_publisher's `--x/--y/--yaw` command line and into
    task1_runner's parameters, and resolving it once here is what keeps those two
    numerically identical. A `LaunchConfiguration` is resolved separately per
    consumer at evaluation time, which is how the start pose came to exist as
    several independent literals to begin with. Resolving here also leaves the
    value readable by anything that introspects the launch graph without running
    it (`ros2 launch --show-args`, the tests in test/).
    """
    prefix = f'{name}:='
    for entry in reversed(argv if argv is not None else sys.argv):
        if entry.startswith(prefix):
            return entry[len(prefix):]
    return default


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

    vision_arg = DeclareLaunchArgument(
        'vision',
        default_value='true',
        description='Bring up camera_node + yolo_detector. Set false for '
                     'sessions that only need drive/steer/telemetry (e.g. '
                     'motor PID bench-tuning) - drops real camera/YOLO CPU '
                     'load on the Pi that a hardware-only session does not '
                     'need. Usage: pixi run real vision:=false'
    )
    vision = LaunchConfiguration('vision')

    # Which task runner (the brain) to bring up. Hardware bringup is identical
    # for both tasks; only the runner differs (task1 = explore + recognise
    # with obstacle setup; task2 = fixed slalom, no setup). Selected here so
    # the same real.launch.py serves both:
    #   pixi run real   -> task:=1   pixi run real2 -> task:=2
    task_arg = DeclareLaunchArgument(
        'task',
        default_value='1',
        description='Which task runner to launch: 1 (explore+recognise) or 2 (slalom).'
    )
    task = LaunchConfiguration('task')

    # Start pose in the arena frame - see DEFAULT_START_* above for why this is
    # declared once and where it is consumed.
    start_pose_args = [
        DeclareLaunchArgument(
            'start_x', default_value=str(DEFAULT_START_X),
            description='Robot rear-axle x in the arena frame at power-on, metres. '
                        'Anywhere inside the 40cm start box is legal - set this to '
                        'where the car actually sits.'),
        DeclareLaunchArgument(
            'start_y', default_value=str(DEFAULT_START_Y),
            description='Robot rear-axle y in the arena frame at power-on, metres.'),
        DeclareLaunchArgument(
            'start_yaw', default_value=str(DEFAULT_START_YAW),
            description='Robot heading in the arena frame at power-on, radians '
                        '(pi/2 = facing arena +Y, "North").'),
    ]
    start_x = float(_launch_arg('start_x', str(DEFAULT_START_X)))
    start_y = float(_launch_arg('start_y', str(DEFAULT_START_Y)))
    start_yaw = float(_launch_arg('start_yaw', str(DEFAULT_START_YAW)))

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

    # ONE spawner for BOTH controllers (not two racing nodes) - see the note
    # in task1_sim.launch.py. Avoids the "Controller already loaded ... Failed
    # to configure" race. (Real uses a standalone ros2_control_node manager;
    # the /cmd_vel remap is already applied there, so not repeated here.)
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'ackermann_steering_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30',
        ],
        output='screen'
    )

    serial_bridge = Node(
        package='mdp_bridge',
        executable='serial_bridge_node',
        parameters=[{'serial_port': serial_port, 'baud_rate': 115200}],
        # The bridge's own JointState publish is TopicBasedSystem's raw
        # hardware-feedback input, not the graph-wide /joint_states topic -
        # joint_state_broadcaster owns that name (see mini_akm_real_robot.urdf's
        # joint_states_topic param). Remapped here rather than in the node's
        # C++ source so the topic name stays a launch-time concern.
        remappings=[('/joint_states', '/joint_states_raw')],
        output='screen'
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[os.path.join(pkg_bringup, 'config', 'ekf.yaml')],
        output='screen'
    )

    # map -> odom: the arena frame's only link to the robot's dead-reckoned pose.
    # `odom` is created at power-on with identity orientation, so the arena pose of
    # `odom` is exactly where the car was placed - a STATIC transform, since with
    # no absolute localization source there is nothing to correct it with and all
    # drift belongs in odom -> base_link.
    #
    # Ownership: this node only. ekf.yaml is untouched - `world_frame` stays `odom`
    # and the EKF keeps sole ownership of odom -> base_link via `publish_tf`.
    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        arguments=[
            '--x', str(start_x), '--y', str(start_y), '--z', '0.0',
            '--yaw', str(start_yaw), '--pitch', '0.0', '--roll', '0.0',
            '--frame-id', 'map', '--child-frame-id', 'odom'
        ],
        output='screen'
    )

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        parameters=[{
            'camera': 0,           # only one camera on this board (RPi Camera Module V2 / IMX219)
            'width': 640,
            'height': 480,
            'format': 'RGB888',    # 3-channel, no alpha - avoids XRGB8888's auto-pick and matches
                                    # cv_bridge's bgr8 conversion in yolo_detector.py cleanly
            'camera_info_url': 'package://mdp_yolo/config/imx219_640x480.yaml',
        }],
        condition=IfCondition(vision),
        output='screen'
    )

    yolo_detector = Node(
        package='mdp_yolo',
        executable='yolo_detector.py',
        parameters=[{'camera_topic': '/camera/image_raw'}],
        condition=IfCondition(vision),
        output='screen'
    )

    # Task 1 runner (task:=1). Comes up idle in WAITING_FOR_SETUP:
    #   pixi run setup -> publishes /obstacle_setup -> it plans, then HOLDS
    #   pixi run go    -> calls /start_run service   -> it starts driving
    # The car does not move on bringup or on setup, only on `go`.
    task1_runner = Node(
        package='mdp_bringup',
        executable='task1_runner.py',
        parameters=[
            os.path.join(pkg_bringup, 'config', 'occupancy_grid_viz.yaml'),
            # Same numbers the map -> odom transform publishes, so the planner
            # starts the route where the car actually stands.
            {'start_x': start_x, 'start_y': start_y, 'start_yaw': start_yaw}
        ],
        condition=IfCondition(EqualsSubstitution(task, '1')),
        output='screen'
    )

    # Task 2 runner (task:=2). Comes up idle in WAITING_FOR_START (no obstacle
    # setup for this task):
    #   pixi run go -> calls /start_run service -> it drives the slalom path.
    task2_runner = Node(
        package='mdp_bringup',
        executable='task2_runner.py',
        condition=IfCondition(EqualsSubstitution(task, '2')),
        output='screen'
    )

    return LaunchDescription([
        serial_port_arg,
        vision_arg,
        task_arg,
        *start_pose_args,
        robot_state_publisher,
        controller_manager,
        controller_spawner,
        serial_bridge,
        ekf_node,
        map_to_odom,
        camera_node,
        yolo_detector,
        task1_runner,
        task2_runner
    ])
