import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mdp_yolo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Install every exported model dir under models/ (best_ncnn_model =
        # MDP-trained default, yolo26n_ncnn_model = stock COCO debug net), each
        # to its own share/mdp_yolo/models/<name>/ so the detector's
        # model_path=<name> switch can resolve any of them.
        *[
            (os.path.join('share', package_name, 'models', os.path.basename(d)),
                [f for f in glob(os.path.join(d, '*')) if os.path.isfile(f)])
            for d in glob('models/*') if os.path.isdir(d)
        ],
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
            'camera_publisher = mdp_yolo.camera_publisher:main',
            'camera_publisher.py = mdp_yolo.camera_publisher:main',
            'yolo_detector = mdp_yolo.yolo_detector:main',
            'yolo_detector.py = mdp_yolo.yolo_detector:main',
        ],
    },
)
