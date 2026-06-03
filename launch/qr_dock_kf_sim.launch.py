"""
qr_dock_kf_sim.launch.py — synthetic closed-loop test (no Gazebo).

Starts the synthetic QR world (qr_sim_world) and the docking node
(qr_dock_kf_node) wired together, then auto-activates docking after a short
delay so you can just watch it dock.

  ros2 launch qr_dock_kf qr_dock_kf_sim.launch.py
  ros2 launch qr_dock_kf qr_dock_kf_sim.launch.py qr_z_tilt_deg:=35.0

View the camera + HUD:
  ros2 run rqt_image_view rqt_image_view /qr_dock_kf_node/debug_image
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('qr_dock_kf')
    calib = os.path.join(pkg, 'config', 'sim_calib.npz')

    qr_z_tilt = LaunchConfiguration('qr_z_tilt_deg')
    qr_size = LaunchConfiguration('qr_size_m')
    robot_x = LaunchConfiguration('robot_x')

    sim = Node(
        package='qr_dock_kf',
        executable='qr_sim_world',
        name='qr_sim_world',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'image_topic': '/image_raw',
            'odom_topic': '/odometry/filtered',
            'cmd_vel_topic': 'alignment_cmd_vel',
            'calib_path': calib,
            'qr_z_tilt_deg': qr_z_tilt,
            'qr_size_m': qr_size,
            'robot_x': robot_x,
        }],
    )

    dock = Node(
        package='qr_dock_kf',
        executable='qr_dock_kf_node',
        name='qr_dock_kf_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'image_topic': '/image_raw',
            'odom_topic': '/odometry/filtered',
            'cmd_vel_topic': 'alignment_cmd_vel',
            'use_odom_twist': True,
            'calib_path': calib,       # SAME intrinsics as the sim camera
            'default_qr_size_mm': 200.0,
            'cam_x_offset_m': 0.14,
            'cam_y_offset_m': 0.0,
            'cam_z_offset_m': 0.205,
            'cam_pitch_deg': 0.0,
            'marker_gap_m': 0.18,
            'waypoint_dist_m': 0.25,
            'loss_timeout_s': 10.0,    # allow coasting through the oblique drive
        }],
    )

    # Auto-activate docking 3 s after start (let the camera stream warm up).
    activate = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '-1', '/align/mode',
                 'std_msgs/msg/String', '{data: dock_qr_kf}'],
            output='screen')],
    )

    return LaunchDescription([
        DeclareLaunchArgument('qr_z_tilt_deg', default_value='25.0'),
        DeclareLaunchArgument('qr_size_m', default_value='0.20'),
        DeclareLaunchArgument('robot_x', default_value='0.0'),
        sim,
        dock,
        activate,
    ])
