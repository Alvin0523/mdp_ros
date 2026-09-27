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
    # mdp_algorithm is a pure planning library - its modules are imported by
    # the task runners in mdp_bringup (e.g. planning.planner,
    # pure_pursuit_follower's PurePursuitController class, spline_planner's
    # SplinePathPlanner class), not launched as standalone nodes. No
    # console_scripts: the previous entries pointed at demo/no-op main()s
    # that nothing ran (and reeds_shepp_planner, now removed, was dead).
    entry_points={
        'console_scripts': [],
    },
)
