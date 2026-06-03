"""
qr_dock_map_sim.launch.py — localization-assisted docking, synthetic test.

Runs the synthetic world (which also publishes a ground-truth map->odom->base
TF tree) and the map-frame docking node, then auto-activates after a delay.

  ros2 launch qr_dock_kf qr_dock_map_sim.launch.py qr_z_tilt_deg:=30.0
  ros2 launch qr_dock_kf qr_dock_map_sim.launch.py robot_x:=-0.3 qr_z_tilt_deg:=20.0
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
        package='qr_dock_kf', executable='qr_sim_world', name='qr_sim_world',
        output='screen', emulate_tty=True,
        parameters=[{
            'calib_path': calib,
            'qr_z_tilt_deg': qr_z_tilt,
            'qr_size_m': qr_size,
            'robot_x': robot_x,
        }],
    )

    dock = Node(
        package='qr_dock_kf', executable='qr_dock_map_node', name='qr_dock_map_node',
        output='screen', emulate_tty=True,
        parameters=[{
            'calib_path': calib,
            'default_qr_size_mm': 200.0,
            'cam_x_offset_m': 0.14,
            'cam_z_offset_m': 0.205,
            'map_frame': 'map',
            'base_frame': 'base_footprint',
            'pregrasp_dist_m': 0.40,
            'marker_gap_m': 0.15,
        }],
    )

    activate = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '-1', '/align/mode',
                 'std_msgs/msg/String', '{data: dock_qr_map}'], output='screen')],
    )

    return LaunchDescription([
        DeclareLaunchArgument('qr_z_tilt_deg', default_value='25.0'),
        DeclareLaunchArgument('qr_size_m', default_value='0.20'),
        DeclareLaunchArgument('robot_x', default_value='0.0'),
        sim, dock, activate,
    ])
