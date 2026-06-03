"""
dock_with_localization.launch.py — END-TO-END on the real Puzzlebot.

Brings up the EKF/MCL/ArUco global localization stack (which publishes the
map -> base_footprint TF) AND the localization-assisted QR docking node.

  ros2 launch qr_dock_kf dock_with_localization.launch.py \
      calib_path:=/home/edrick/Github/TCSM/src/tcsm_camera_utils/data/calibration_data/calibration_data_3_PERFECT.npz

Then trigger docking:
  ros2 topic pub -1 /align/mode std_msgs/msg/String "{data: dock_qr_map}"
  # or: ros2 service call /qr_dock_map/enable custom_interfaces/srv/SetProcessBool "{enable: true}"

IMPORTANT real-robot notes:
  * The QR node detects on a RAW sensor_msgs/Image (image_topic, default
    /image_raw). The ArUco localizer uses the compressed stream separately;
    both come from the same camera.
  * cam_*_offset_m / cam_pitch_deg MUST match your real camera mount (MEASURE
    them). The repo has conflicting values across nodes; defaults here follow
    qr_pose_align's measured 5cm fwd / 20.5cm up.
  * pregrasp_dist_m is the closest distance at which the node still needs to SEE
    the QR (REFINE). Everything closer is open-loop COMMIT. Raise it if your QR
    is hard to detect up close.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sim = get_package_share_directory('tcsm_sim')

    calib_path = LaunchConfiguration('calib_path')
    image_topic = LaunchConfiguration('image_topic')
    use_compressed_image = LaunchConfiguration('use_compressed_image')
    use_rviz = LaunchConfiguration('use_rviz')

    # ---- Global localization (EKF + MCL + ArUco) -> TF map->base_footprint ----
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sim, 'launch', 'tcsm_ekf_mcl_aruco.launch.py')),
        launch_arguments={'use_rviz': use_rviz}.items(),
    )

    # ---- Localization-assisted QR docking ----
    dock = Node(
        package='qr_dock_kf',
        executable='qr_dock_map_node',
        name='qr_dock_map_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'image_topic': image_topic,
            'use_compressed_image': use_compressed_image,
            'odom_topic': '/odometry/filtered',
            'calib_path': calib_path,
            'map_frame': 'map',
            'base_frame': 'base_footprint',
            # MEASURE these on the real robot:
            'cam_x_offset_m': 0.05,
            'cam_y_offset_m': 0.0,
            'cam_z_offset_m': 0.205,
            'cam_pitch_deg': 0.0,
            'default_qr_size_mm': 97.0,
            'pregrasp_dist_m': 0.45,
            'marker_gap_m': 0.15,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('calib_path', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument(
            'use_compressed_image', default_value='false',
            description='If true, image_topic is a sensor_msgs/CompressedImage '
                        '(e.g. /image_raw/compressed).'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        localization,
        dock,
    ])
