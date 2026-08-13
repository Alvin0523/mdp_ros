import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


ARGUMENTS = [
    DeclareLaunchArgument(
        'use_rviz', default_value='true',
        choices=['true', 'false'],
        description='Launch RViz'),
]


def generate_launch_description():
    pkg_mdp_description = get_package_share_directory('mdp_description')

    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_mdp_description, 'launch', 'display.launch.py')),
        launch_arguments={
            'use_joint_state_gui': 'false',
            'use_rviz': LaunchConfiguration('use_rviz'),
        }.items(),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(display_launch)
    return ld
