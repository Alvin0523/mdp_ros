from glob import glob

from setuptools import find_packages, setup

package_name = 'mdp_bringup'

# Nodes the launch starts, then the one-shot command-line tools (pixi run go /
# setup / calib ...). `ros2 run mdp_bringup <name>`.
NODES = ['task1_runner', 'task2_runner', 'robot_pose_feedback', 'manual_drive',
         'bt_monitor', 'health_monitor', 'sim_obstacles']
TOOLS = {'trigger': 'trigger', 'publish_obstacles': 'publish_obstacles',
         'around_obstacle': 'around_obstacle', 'calib': 'calib.cli'}   # command -> module

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wm_u26',
    maintainer_email='alvinwm0523@gmail.com',
    description='Launch, config and task nodes for the MDP car',
    license='TODO',
    entry_points={
        'console_scripts':
            [f'{n} = mdp_bringup.{n}:main' for n in NODES] +
            [f'{n} = mdp_bringup.tools.{m}:main' for n, m in TOOLS.items()],
    },
)
