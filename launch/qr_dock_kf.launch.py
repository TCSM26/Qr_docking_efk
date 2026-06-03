"""
qr_dock_kf.launch.py — Kalman-filtered QR docking node.

Standalone; does NOT launch or touch the EKF / MCL / ArUco stack. Run it
alongside your localization stack. Activate docking by publishing
  ros2 topic pub -1 /align/mode std_msgs/String "data: dock_qr_kf"
or by calling the enable service
  ros2 service call /qr_dock_kf/enable custom_interfaces/srv/SetProcessBool "{enable: true}"

Override the calibration path for real runs, e.g.:
  ros2 launch qr_dock_kf qr_dock_kf.launch.py \
      calib_path:=/home/edrick/Github/TCSM/src/tcsm_camera_utils/data/calibration_data/calibration_data_3_PERFECT.npz
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('qr_dock_kf')
    default_params = os.path.join(pkg, 'config', 'qr_dock_kf.yaml')

    params_file = LaunchConfiguration('params_file')
    image_topic = LaunchConfiguration('image_topic')
    calib_path = LaunchConfiguration('calib_path')

    node = Node(
        package='qr_dock_kf',
        executable='qr_dock_kf_node',
        name='qr_dock_kf_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            params_file,
            {
                'image_topic': image_topic,
                'calib_path': calib_path,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('calib_path', default_value=''),
        node,
    ])
