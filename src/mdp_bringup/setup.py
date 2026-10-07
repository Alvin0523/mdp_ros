from glob import glob

from setuptools import find_packages, setup

package_name = 'mdp_bringup'

# Nodes the launch starts (command -> module), then the one-shot command-line
# tools (pixi run go / setup / calib ...). `ros2 run mdp_bringup <command>`.
NODES = {
    'task1_runner': 'tasks.task1_runner', 'task2_runner': 'tasks.task2_runner',
    'task1_planner': 'tasks.task1_planner',
    'robot_pose_feedback': 'robot.robot_pose_feedback', 'manual_drive': 'robot.manual_drive',
    'health_monitor': 'robot.health_monitor', 'bag_recorder': 'robot.bag_recorder',
    'bt_monitor': 'robot.bt_monitor', 'pi_status': 'robot.pi_status',
    'sim_helpers': 'sim.sim_helpers',
}
TOOLS = {'trigger': 'tools.trigger', 'publish_obstacles': 'tools.publish_obstacles',
         'around_obstacle': 'tools.around_obstacle', 'calib': 'tools.calib.cli',
         'capture': 'tools.capture'}

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
            [f'{n} = mdp_bringup.{m}:main' for n, m in {**NODES, **TOOLS}.items()],
    },
)
