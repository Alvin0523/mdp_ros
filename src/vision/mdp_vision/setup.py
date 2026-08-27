import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mdp_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MDP Team',
    maintainer_email='user@todo.todo',
    description='Vision stack for MDP robot',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'camera_publisher = mdp_vision.camera_publisher:main',
            'camera_publisher.py = mdp_vision.camera_publisher:main',
            'yolo_arrow_detector = mdp_vision.yolo_arrow_detector:main',
            'yolo_arrow_detector.py = mdp_vision.yolo_arrow_detector:main',
        ],
    },
)
