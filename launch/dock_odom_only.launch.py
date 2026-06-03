"""
dock_odom_only.launch.py — MINIMAL real-robot docking test (NO ArUcos / map / lidar).

For a first test on the Puzzlebot you do NOT need the ArUco track. The docking
node only needs a map->base_footprint TF; here that TF is pure wheel odometry
(map == odom, a static identity). Because the docking maneuver is short (~1 m)
and the architecture is robust to localization drift (the REFINE visual
correction absorbs it), odometry-only is enough to validate docking on the
robot with just: wheel encoders + camera + a QR on a wall.

Requires the robot base to publish wheel speeds on VelocityEncL / VelocityEncR
(Float32) and a camera on image_topic.

  ros2 launch qr_dock_kf dock_odom_only.launch.py \
      calib_path:=/home/edrick/Github/TCSM/src/tcsm_camera_utils/data/calibration_data/calibration_data_3_PERFECT.npz

  ros2 topic pub -1 /align/mode std_msgs/msg/String "{data: dock_qr_map}"

When you later want true global localization (recover from drift across a large
space), switch to dock_with_localization.launch.py (EKF/MCL/ArUco + the track).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    calib_path = LaunchConfiguration('calib_path')
    image_topic = LaunchConfiguration('image_topic')
    use_compressed_image = LaunchConfiguration('use_compressed_image')

    # Wheel odometry -> /odometry/filtered + TF odom->base_footprint.
    # Use YOUR calibrated wheel values.
    odometry = Node(
        package='movement_control',
        executable='odometry_node',
        name='odometry',
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'sample_time': 0.018,
            'odom_topic': '/odometry/filtered',
            'publish_tf': True,
        }],
    )

    # No global localization: map == odom (identity). Dead-reckoning is enough
    # for the short docking maneuver.
    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_identity',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )

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
        DeclareLaunchArgument('use_compressed_image', default_value='false'),
        odometry,
        map_to_odom,
        dock,
    ])
