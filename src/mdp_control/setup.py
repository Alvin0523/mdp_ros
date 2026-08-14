from setuptools import find_packages, setup

package_name = 'mdp_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MDP Team',
    maintainer_email='user@todo.todo',
    description='Autonomy, Vision, and Path Planning nodes for Mini Ackermann Robot',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'task1_runner = mdp_control.task1_runner:main',
            'task1_runner.py = mdp_control.task1_runner:main',
            'task2_runner = mdp_control.task2_runner:main',
            'task2_runner.py = mdp_control.task2_runner:main',
            'pure_pursuit_follower = mdp_control.pure_pursuit_follower:main',
            'pure_pursuit_follower.py = mdp_control.pure_pursuit_follower:main',
            'reeds_shepp_planner = mdp_control.reeds_shepp_planner:main',
            'reeds_shepp_planner.py = mdp_control.reeds_shepp_planner:main',
            'spline_planner = mdp_control.spline_planner:main',
            'spline_planner.py = mdp_control.spline_planner:main',
        ],
    },
)
