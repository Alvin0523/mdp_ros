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
        # PyTorch weights (best_v5.pt, best_v4.pt, ...), one file each:
        # share/mdp_vision/models/<name>.pt - the detector's model_path=<name> switch.
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ] + [
        # CoreML exports (Mac, `pixi run export-coreml`): <name>.mlpackage is a
        # folder - every file in it, keeping its layout.
        (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
        for pkg in glob('models/*.mlpackage') for root, _, files in os.walk(pkg) if files
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
            'rpi_cam_publisher = mdp_vision.rpi_cam_publisher:main',
            'yolo_detector = mdp_vision.yolo_detector:main',
            'bullseye_matcher = mdp_vision.bullseye_matcher:main',
            'camera_relay = mdp_vision.camera_relay:main',
        ],
    },
)
