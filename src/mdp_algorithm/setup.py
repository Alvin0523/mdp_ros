from setuptools import find_packages, setup

package_name = 'mdp_algorithm'

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
    description='Path planning building blocks for the Mini Ackermann Robot',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'pure_pursuit_follower = mdp_algorithm.pure_pursuit_follower:main',
            'pure_pursuit_follower.py = mdp_algorithm.pure_pursuit_follower:main',
            'reeds_shepp_planner = mdp_algorithm.reeds_shepp_planner:main',
            'reeds_shepp_planner.py = mdp_algorithm.reeds_shepp_planner:main',
            'spline_planner = mdp_algorithm.spline_planner:main',
            'spline_planner.py = mdp_algorithm.spline_planner:main',
        ],
    },
)
