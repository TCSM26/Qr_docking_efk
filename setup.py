import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'qr_dock_kf'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='edrick',
    maintainer_email='edrickgv05@gmail.com',
    description='Kalman-filtered QR docking (ego-motion prediction + waypoint state machine) for the TCSM Puzzlebot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qr_dock_kf_node = qr_dock_kf.qr_dock_kf_node:main',
            'qr_dock_map_node = qr_dock_kf.qr_dock_map_node:main',
            'qr_sim_world = qr_dock_kf.qr_sim_world:main',
        ],
    },
)
