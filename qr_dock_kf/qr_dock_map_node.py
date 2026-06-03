#!/usr/bin/env python3
"""
qr_dock_map_node.py
===================
Localization-assisted QR docking for the TCSM Puzzlebot.

Strategy (chosen after sim benchmarking showed that *pure visual* docking
from an oblique start is fragile -- the QR leaves the FOV mid-approach and the
recovery diverges, regardless of the controller):

  OBSERVE : while the robot is globally localized (EKF/MCL/ArUco -> TF
            map->base_footprint), detect the fixed QR, transform its pose into
            the MAP frame and lock it once stable. The QR is seen from afar so
            it stays small/centered in the frame -> reliable.
  NAV_MAP : drive to a HEAD-ON pre-grasp pose placed on the QR's outward normal,
            using the global map pose as feedback (rho-alpha-beta). This big
            maneuver does NOT use vision, so it does not matter if the QR
            leaves the FOV.
  REFINE  : at the pre-grasp (head-on, QR small and centered) re-acquire the QR
            and do a short visual centering/perpendicularity correction to
            absorb localization error.
  COMMIT  : open-loop straight advance by odometry to the final standoff.

Does NOT modify the EKF / MCL / ArUco stack; it only consumes their TF and
publishes to alignment_cmd_vel (merged by cmd_vel_mux), like the other aligners.
"""

import math
import os
import re
from enum import Enum

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from std_msgs.msg import Bool, String, ColorRGBA, Int32
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor
from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped transform support)

try:
    from custom_interfaces.srv import SetProcessBool
except ImportError:
    SetProcessBool = None


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quat(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def build_cam_extrinsics(x, y, z, pitch_rad):
    s, c = math.sin(pitch_rad), math.cos(pitch_rad)
    R = np.array([[0.0, -s, c], [-1.0, 0.0, 0.0], [0.0, -c, -s]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


class Phase(str, Enum):
    IDLE = "idle"
    OBSERVE = "observe"     # lock the QR pose in the map frame
    NAV_MAP = "nav_map"     # drive to head-on pre-grasp by localization
    REFINE = "refine"       # short visual centering at the pre-grasp
    COMMIT = "commit"       # open-loop final advance
    DONE = "done"


class QRDockMapNode(Node):
    def __init__(self):
        super().__init__("qr_dock_map_node")

        # ---- I/O ----
        # QR pose source:
        #   "internal" -> detect & solvePnP on the image here (default; sim).
        #   "external" -> consume the on-Jetson /qr/pose (full-res detection),
        #                 transformed via TF. Better and offloads the PC.
        self.declare_parameter("qr_pose_source", "internal")
        self.declare_parameter("qr_pose_topic", "/qr/pose")
        self.declare_parameter("qr_data_topic", "/qr/data")
        self.declare_parameter("target_qr_id", "",        # "" = accept any QR
                               ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("qr_normal_axis", "x")     # QR outward-face axis
        self.declare_parameter("manage_qr_enable", True)  # toggle /qr_enable
        self.declare_parameter("qr_enable_topic", "/qr_enable")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("use_compressed_image", False)
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("cmd_vel_topic", "alignment_cmd_vel")
        self.declare_parameter("done_topic", "/align/done")
        self.declare_parameter("mode_topic", "/align/mode")
        self.declare_parameter("activate_mode", "dock_qr_map")
        self.declare_parameter("enable_service", "/qr_dock_map/enable")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        # ---- camera / QR ----
        self.declare_parameter("calib_path", "")
        self.declare_parameter("default_qr_size_mm", 97.0)
        self.declare_parameter("qr_size_overrides", "")
        self.declare_parameter("cam_x_offset_m", 0.14)
        self.declare_parameter("cam_y_offset_m", 0.0)
        self.declare_parameter("cam_z_offset_m", 0.205)
        self.declare_parameter("cam_pitch_deg", 0.0)
        self.declare_parameter("max_reproj_px", 6.0)

        # ---- geometry ----
        self.declare_parameter("pregrasp_dist_m", 0.40)   # head-on standoff (QR visible)
        self.declare_parameter("marker_gap_m", 0.15)      # final stop distance

        # ---- OBSERVE ----
        self.declare_parameter("observe_frames", 12)
        self.declare_parameter("observe_pos_std_m", 0.03)
        self.declare_parameter("observe_yaw_std_rad", 0.08)
        self.declare_parameter("observe_search", True)    # rotate to find QR
        self.declare_parameter("observe_search_speed", 0.15)
        self.declare_parameter("observe_timeout_s", 30.0)

        # ---- NAV_MAP (rho-alpha-beta in map frame) ----
        self.declare_parameter("k_rho", 0.5)
        self.declare_parameter("k_alpha", 1.2)
        self.declare_parameter("k_beta", -0.35)
        self.declare_parameter("nav_tol_xy_m", 0.04)
        self.declare_parameter("nav_tol_yaw_deg", 6.0)
        self.declare_parameter("nav_stable_s", 0.4)
        self.declare_parameter("nav_timeout_s", 60.0)

        # ---- REFINE ----
        self.declare_parameter("refine_kp_ang", 1.2)
        self.declare_parameter("refine_tol_lat_m", 0.015)
        self.declare_parameter("refine_stable_s", 0.4)
        self.declare_parameter("refine_timeout_s", 4.0)   # commit anyway if QR not re-acquired

        # ---- COMMIT ----
        self.declare_parameter("commit_speed", 0.04)
        self.declare_parameter("commit_max_dist_m", 0.40)

        # ---- limits / misc ----
        self.declare_parameter("max_lin_speed", 0.10)
        self.declare_parameter("max_ang_speed", 0.6)
        self.declare_parameter("min_lin_speed", 0.0)
        self.declare_parameter("min_ang_speed", 0.02)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_rate_hz", 10.0)
        self.declare_parameter("debug", False)
        self.declare_parameter("start_enabled", False)

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g("image_topic")
        self.use_compressed = bool(g("use_compressed_image"))
        self.qr_pose_source = str(g("qr_pose_source")).lower()
        self.qr_pose_topic = g("qr_pose_topic")
        self.qr_data_topic = g("qr_data_topic")
        self.target_qr_id = str(g("target_qr_id")).strip()
        self.qr_normal_axis = str(g("qr_normal_axis")).lower()
        self.manage_qr_enable = bool(g("manage_qr_enable"))
        self.qr_enable_topic = g("qr_enable_topic")
        self.last_qr_payload_id = None
        self.odom_topic = g("odom_topic")
        self.cmd_vel_topic = g("cmd_vel_topic")
        self.done_topic = g("done_topic")
        self.mode_topic = g("mode_topic")
        self.activate_mode = g("activate_mode")
        self.enable_service = g("enable_service")
        self.map_frame = g("map_frame")
        self.base_frame = g("base_frame")

        self.default_qr_size_m = g("default_qr_size_mm") / 1000.0
        self.size_overrides = self._parse_overrides(g("qr_size_overrides"))
        self.T_base_cam = build_cam_extrinsics(
            g("cam_x_offset_m"), g("cam_y_offset_m"), g("cam_z_offset_m"),
            math.radians(g("cam_pitch_deg")))
        self.max_reproj_px = float(g("max_reproj_px"))

        self.d_pre = float(g("pregrasp_dist_m"))
        self.marker_gap = float(g("marker_gap_m"))

        self.observe_frames = int(g("observe_frames"))
        self.observe_pos_std = float(g("observe_pos_std_m"))
        self.observe_yaw_std = float(g("observe_yaw_std_rad"))
        self.observe_search = bool(g("observe_search"))
        self.observe_search_speed = float(g("observe_search_speed"))
        self.observe_timeout = float(g("observe_timeout_s"))

        self.k_rho = float(g("k_rho"))
        self.k_alpha = float(g("k_alpha"))
        self.k_beta = float(g("k_beta"))
        self.nav_tol_xy = float(g("nav_tol_xy_m"))
        self.nav_tol_yaw = math.radians(g("nav_tol_yaw_deg"))
        self.nav_stable = float(g("nav_stable_s"))
        self.nav_timeout = float(g("nav_timeout_s"))

        self.refine_kp_ang = float(g("refine_kp_ang"))
        self.refine_tol_lat = float(g("refine_tol_lat_m"))
        self.refine_stable = float(g("refine_stable_s"))
        self.refine_timeout = float(g("refine_timeout_s"))

        self.commit_speed = float(g("commit_speed"))
        self.commit_max_dist = float(g("commit_max_dist_m"))

        self.v_max = float(g("max_lin_speed"))
        self.w_max = float(g("max_ang_speed"))
        self.v_min = float(g("min_lin_speed"))
        self.w_min = float(g("min_ang_speed"))
        self.rate_hz = float(g("control_rate_hz"))
        self.publish_debug = bool(g("publish_debug_image"))
        self.debug_period = 1.0 / max(0.1, g("debug_image_rate_hz"))
        self.debug = bool(g("debug"))

        (self.K, self.dist, self.cam_model) = self._load_calibration(g("calib_path"))

        # ---- state ----
        self.enabled = bool(g("start_enabled"))
        self.phase = Phase.IDLE
        self.obs = []                 # accumulated QR map poses (x,y,yaw)
        self.qr_map = None            # locked (qx, qy, qyaw) in map
        self.pre = None               # pre-grasp (px, py, pyaw) in map
        self.last_qr_base = None      # latest visual (mx, my, nyaw) in base
        self.last_qr_t = None
        self.phase_t0 = None
        self.in_tol_since = None
        self.commit_start = None
        self.commit_target = 0.0
        self.done_latched = False
        self.odom_px = 0.0
        self.odom_py = 0.0
        self.last_corners = None
        self.last_payload = ""
        self.last_debug_t = 0.0

        self.bridge = CvBridge()
        self.qr_det = cv2.QRCodeDetector()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_qr_enable = None
        if self.qr_pose_source == "external":
            # Consume the on-Jetson full-res QR pose; no local image processing.
            self.create_subscription(PoseStamped, self.qr_pose_topic,
                                     self._on_qr_pose, sensor_qos)
            self.create_subscription(String, self.qr_data_topic,
                                     self._on_qr_data, 10)
            if self.manage_qr_enable:
                self.pub_qr_enable = self.create_publisher(
                    Int32, self.qr_enable_topic, 10)
        elif self.use_compressed:
            self.create_subscription(CompressedImage, self.image_topic,
                                     self._on_image_compressed, sensor_qos)
        else:
            self.create_subscription(Image, self.image_topic,
                                     self._on_image, sensor_qos)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.create_subscription(String, self.mode_topic, self._on_mode, 10)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_done = self.create_publisher(Bool, self.done_topic, 10)
        self.pub_dbg = self.create_publisher(Image, "~/debug_image", 10)
        self.pub_qr = self.create_publisher(Marker, "~/qr_marker", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "~/pregrasp", 10)

        if SetProcessBool is not None:
            self.create_service(SetProcessBool, self.enable_service, self._on_enable)

        self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f"qr_dock_map ready. activate via /align/mode='{self.activate_mode}'. "
            f"map_frame={self.map_frame} base_frame={self.base_frame}")

    # ------------------------------------------------------------------ #
    def _parse_overrides(self, s):
        out = {}
        for item in (s or "").split(","):
            item = item.strip()
            if "=" in item:
                k, _, v = item.rpartition("=")
                try:
                    out[k.strip()] = float(v) / 1000.0
                except ValueError:
                    pass
        return out

    def _qr_size_for(self, content):
        if content:
            m = re.search(r"qr_mm\s*=\s*([0-9.]+)", content)
            if m:
                try:
                    return float(m.group(1)) / 1000.0
                except ValueError:
                    pass
            for k, v in self.size_overrides.items():
                if k in content:
                    return v
        return self.default_qr_size_m

    def _load_calibration(self, path):
        if path and os.path.isfile(path):
            data = np.load(path, allow_pickle=True)
            K = data["K"] if "K" in data.files else data["mtx"]
            D = data["dist"] if "dist" in data.files else data["D"]
            model = str(data["model"]) if "model" in data.files else "pinhole"
            return (K.astype(np.float64),
                    np.asarray(D, dtype=np.float64).reshape(-1, 1), model)
        fx = fy = 320.0 / (2 * math.tan(math.radians(30)))
        return np.array([[fx, 0, 160.0], [0, fy, 120.0], [0, 0, 1.0]]), \
            np.zeros((5, 1)), "pinhole"

    # ------------------------------------------------------------------ #
    def _start(self):
        self.enabled = True
        self.phase = Phase.OBSERVE
        self.obs = []
        self.qr_map = None
        self.pre = None
        self.last_qr_base = None
        self.last_qr_t = None
        self.phase_t0 = self._now()
        self.in_tol_since = None
        self.commit_start = None
        self.done_latched = False
        self.last_qr_payload_id = None
        self._publish_zero()
        self._set_qr_enable(1)          # turn on the on-Jetson QR detection
        self.get_logger().info("qr_dock_map ENABLED -> OBSERVE")

    def _stop(self):
        self.enabled = False
        self.phase = Phase.IDLE
        self._publish_zero()
        self._set_qr_enable(0)          # save the Jetson some compute

    def _set_qr_enable(self, on):
        if self.pub_qr_enable is not None:
            self.pub_qr_enable.publish(Int32(data=int(on)))

    def _on_enable(self, req, resp):
        self._start() if bool(req.enable) else self._stop()
        resp.success = True
        resp.message = "enabled" if self.enabled else "disabled"
        return resp

    def _on_mode(self, msg):
        if msg.data.strip() == self.activate_mode:
            if not self.enabled:
                self._start()
        elif self.enabled:
            self._stop()

    def _on_odom(self, msg):
        self.odom_px = msg.pose.pose.position.x
        self.odom_py = msg.pose.pose.position.y

    def _robot_in_map(self):
        """Look up the global robot pose (rx, ry, ryaw) from TF, or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        tr = t.transform.translation
        q = t.transform.rotation
        return tr.x, tr.y, yaw_from_quat(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------ #
    #  Vision: QR -> base_link, and (during OBSERVE) -> map
    # ------------------------------------------------------------------ #
    def _on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}", throttle_duration_sec=2.0)
            return
        self._process_frame(frame)

    def _on_image_compressed(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge(compressed): {e}", throttle_duration_sec=2.0)
            return
        self._process_frame(frame)

    def _process_frame(self, frame):
        base = self._detect_qr_base(frame)
        self._maybe_debug(frame)
        if base is None:
            return
        self._on_qr_base(*base)

    def _on_qr_base(self, mx, my, nyaw):
        """Common handler: QR pose (x, y, outward-normal yaw) in base_link.
        Fed by both the internal detector and the external /qr/pose path."""
        self.last_qr_base = (mx, my, nyaw)
        self.last_qr_t = self._now()

        if self.phase == Phase.OBSERVE:
            rb = self._robot_in_map()
            if rb is None:
                self.get_logger().warn("no map->base TF yet (localization up?)",
                                       throttle_duration_sec=2.0)
                return
            rx, ry, rth = rb
            c, s = math.cos(rth), math.sin(rth)
            qx = rx + c * mx - s * my
            qy = ry + s * mx + c * my
            qyaw = wrap_to_pi(rth + nyaw)
            self.obs.append((qx, qy, qyaw))

    # ---- external on-Jetson /qr/pose path -------------------------------- #
    def _on_qr_data(self, msg):
        self.last_qr_payload_id = self._parse_qr_id(msg.data)

    def _on_qr_pose(self, msg):
        # Optional filter by payload id (e.g. the dock QR specifically).
        if self.target_qr_id and self.last_qr_payload_id != self.target_qr_id:
            return
        # Transform the camera-frame QR pose into base_link via TF.
        try:
            p_base = self.tf_buffer.transform(
                msg, self.base_frame, timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f"TF {msg.header.frame_id}->{self.base_frame}: {e}",
                                   throttle_duration_sec=2.0)
            return
        mx = p_base.pose.position.x
        my = p_base.pose.position.y
        q = p_base.pose.orientation
        nx, ny = self._quat_axis_xy(q, self.qr_normal_axis)
        # The outward normal must point back toward the robot (-x in base);
        # flip if the chosen axis points the other way.
        if nx > 0.0:
            nx, ny = -nx, -ny
        self._on_qr_base(float(mx), float(my), math.atan2(ny, nx))

    @staticmethod
    def _parse_qr_id(text):
        if not text:
            return None
        m = re.search(r'"id"\s*:\s*"?([^",}\s]+)', text)   # JSON {"id": N}
        return m.group(1) if m else text.strip()

    @staticmethod
    def _quat_axis_xy(q, axis):
        """XY components of the QR's outward-face axis ('x' or 'z') in the
        target frame, given the QR orientation quaternion."""
        x, y, z, w = q.x, q.y, q.z, q.w
        if axis == "z":          # 3rd column of R
            return 2 * (x * z + y * w), 2 * (y * z - x * w)
        return 1 - 2 * (y * y + z * z), 2 * (x * y + z * w)   # 'x': 1st column

    def _detect_qr_base(self, frame):
        data, points, _ = self.qr_det.detectAndDecode(frame)
        if points is None or len(points) == 0:
            self.last_corners = None
            return None
        corners = np.asarray(points).reshape(-1, 2).astype(np.float32)
        if corners.shape[0] != 4:
            self.last_corners = None
            return None
        self.last_corners = corners
        self.last_payload = data or self.last_payload
        size = self._qr_size_for(data)
        h = size / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
        cpx = corners.reshape(-1, 1, 2).astype(np.float64)
        if self.cam_model == "fisheye":
            cpx = cv2.fisheye.undistortPoints(cpx, self.K, self.dist, P=self.K)
            dist = np.zeros((4, 1))
        else:
            dist = self.dist
        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                obj, cpx.reshape(-1, 2).astype(np.float32), self.K, dist,
                flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not n:
            return None
        cands = []
        for i in range(n):
            R, _ = cv2.Rodrigues(rvecs[i])
            ncam = R @ np.array([0.0, 0.0, 1.0])
            cands.append((R, tvecs[i].flatten(), ncam,
                          float(reproj[i]) if reproj is not None else 0.0))
        # marker faces the camera -> outward normal has n_cam_z < 0
        pool = [c for c in cands if c[2][2] < 0.0] or cands
        R, t, ncam, e = min(pool, key=lambda c: c[3])
        if e > self.max_reproj_px:
            return None
        pos = (self.T_base_cam @ np.append(t, 1.0))[:3]
        nbase = self.T_base_cam[:3, :3] @ ncam
        return float(pos[0]), float(pos[1]), math.atan2(nbase[1], nbase[0])

    # ------------------------------------------------------------------ #
    #  Control / state machine
    # ------------------------------------------------------------------ #
    def _control_loop(self):
        if not self.enabled or self.phase in (Phase.IDLE, Phase.DONE):
            if self.phase == Phase.DONE:
                self._publish_zero()
            return
        now = self._now()

        if self.phase == Phase.OBSERVE:
            self._tick_observe(now)
        elif self.phase == Phase.NAV_MAP:
            self._tick_nav(now)
        elif self.phase == Phase.REFINE:
            self._tick_refine(now)
        elif self.phase == Phase.COMMIT:
            self._tick_commit()

    def _tick_observe(self, now):
        # Lock the QR map pose once we have enough stable observations.
        if len(self.obs) >= self.observe_frames:
            arr = np.array(self.obs[-self.observe_frames:])
            pos_std = float(np.linalg.norm(np.std(arr[:, :2], axis=0)))
            s = np.mean(np.sin(arr[:, 2]))
            c = np.mean(np.cos(arr[:, 2]))
            yaw_std = math.sqrt(-2.0 * math.log(max(1e-9, min(1.0, math.hypot(s, c)))))
            if pos_std <= self.observe_pos_std and yaw_std <= self.observe_yaw_std:
                qx = float(arr[:, 0].mean())
                qy = float(arr[:, 1].mean())
                qyaw = math.atan2(s, c)
                self.qr_map = (qx, qy, qyaw)
                self.pre = (qx + self.d_pre * math.cos(qyaw),
                            qy + self.d_pre * math.sin(qyaw),
                            wrap_to_pi(qyaw + math.pi))
                self._publish_qr_marker()
                self.phase = Phase.NAV_MAP
                self.phase_t0 = now
                self.in_tol_since = None
                self.get_logger().info(
                    f"QR locked in map: ({qx:.2f},{qy:.2f},{math.degrees(qyaw):.0f}deg) "
                    f"-> pre-grasp ({self.pre[0]:.2f},{self.pre[1]:.2f}). NAV_MAP.")
                return
            # drifting -> drop the oldest half and keep watching
            self.obs = self.obs[len(self.obs) // 2:]

        # Not enough yet: hold, or slowly rotate to bring the QR into view.
        seen_recently = self.last_qr_t is not None and (now - self.last_qr_t) < 0.5
        if self.observe_search and not seen_recently:
            self._publish_cmd(0.0, self.observe_search_speed)
        else:
            self._publish_zero()
        if (now - self.phase_t0) > self.observe_timeout:
            self.get_logger().warn("OBSERVE timeout; aborting.")
            self._stop()

    def _tick_nav(self, now):
        rb = self._robot_in_map()
        if rb is None:
            self._publish_zero()
            return
        rx, ry, rth = rb
        gx, gy, gth = self.pre
        self._publish_pregrasp()

        dx, dy = gx - rx, gy - ry
        rho = math.hypot(dx, dy)
        gamma = math.atan2(dy, dx)
        alpha = wrap_to_pi(gamma - rth)
        beta = wrap_to_pi(gth - gamma)
        yaw_err = wrap_to_pi(gth - rth)

        if rho < self.nav_tol_xy and abs(yaw_err) < self.nav_tol_yaw:
            if self.in_tol_since is None:
                self.in_tol_since = now
            elif (now - self.in_tol_since) >= self.nav_stable:
                self.phase = Phase.REFINE
                self.phase_t0 = now
                self.in_tol_since = None
                self.get_logger().info("NAV_MAP done (head-on pre-grasp). REFINE.")
            self._publish_zero()
            return
        self.in_tol_since = None

        if rho < self.nav_tol_xy:
            # arrived in position, just fix heading to gth
            w = math.copysign(min(self.w_max, 1.5 * abs(yaw_err)), yaw_err)
            self._publish_cmd(0.0, w)
        elif abs(alpha) > math.pi / 2:
            w = math.copysign(min(self.w_max, 1.2 * abs(alpha)), alpha)
            self._publish_cmd(0.0, w)
        else:
            v = self.k_rho * rho * max(0.0, math.cos(alpha))
            w = self.k_alpha * alpha + self.k_beta * beta
            self._publish_cmd(v, w)

        if (now - self.phase_t0) > self.nav_timeout:
            self.get_logger().warn("NAV_MAP timeout; aborting.")
            self._stop()

    def _tick_refine(self, now):
        fresh = self.last_qr_t is not None and (now - self.last_qr_t) < 0.4
        if not fresh:
            # QR not re-acquired (yet). Wait briefly, then trust localization
            # and commit anyway.
            self._publish_zero()
            if (now - self.phase_t0) > self.refine_timeout:
                self.get_logger().warn("REFINE: QR not re-acquired; commit on localization.")
                self._enter_commit(now, visual=False)
            return
        mx, my, _ = self.last_qr_base
        bearing = math.atan2(my, mx)
        if abs(my) <= self.refine_tol_lat:
            if self.in_tol_since is None:
                self.in_tol_since = now
            elif (now - self.in_tol_since) >= self.refine_stable:
                self._enter_commit(now, visual=True)
            self._publish_zero()
            return
        self.in_tol_since = None
        w = self.refine_kp_ang * bearing
        self._publish_cmd(0.0, w)

    def _enter_commit(self, now, visual):
        if visual and self.last_qr_base is not None:
            remaining = self.last_qr_base[0] - self.marker_gap
        else:
            remaining = self.d_pre - self.marker_gap
        self.commit_target = float(np.clip(remaining, 0.0, self.commit_max_dist))
        self.commit_start = (self.odom_px, self.odom_py)
        self.in_tol_since = None
        self.phase = Phase.COMMIT
        self.get_logger().info(
            f"REFINE -> COMMIT: advance {self.commit_target:.3f} m open-loop.")

    def _tick_commit(self):
        traveled = math.hypot(self.odom_px - self.commit_start[0],
                              self.odom_py - self.commit_start[1])
        if traveled >= self.commit_target:
            self._publish_zero()
            self.phase = Phase.DONE
            if not self.done_latched:
                self.pub_done.publish(Bool(data=True))
                self.done_latched = True
                self._set_qr_enable(0)
                self.get_logger().info(f"COMMIT done ({traveled:.3f} m). /align/done=true.")
            return
        self._publish_cmd(self.commit_speed, 0.0)

    # ------------------------------------------------------------------ #
    def _saturate(self, v, w):
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))
        if abs(v) < self.v_min:
            v = 0.0
        if abs(w) < self.w_min:
            w = 0.0
        return v, w

    def _publish_cmd(self, v, w):
        v, w = self._saturate(v, w)
        m = Twist()
        m.linear.x, m.angular.z = float(v), float(w)
        self.pub_cmd.publish(m)

    def _publish_zero(self):
        self.pub_cmd.publish(Twist())

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ #
    def _publish_pregrasp(self):
        if self.pre is None or self.pub_goal.get_subscription_count() == 0:
            return
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.map_frame
        ps.pose.position.x, ps.pose.position.y = self.pre[0], self.pre[1]
        ps.pose.orientation = yaw_to_quat(self.pre[2])
        self.pub_goal.publish(ps)

    def _publish_qr_marker(self):
        if self.qr_map is None or self.pub_qr.get_subscription_count() == 0:
            return
        mk = Marker()
        mk.header.frame_id = self.map_frame
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "qr_dock_map"
        mk.id = 0
        mk.type = Marker.CUBE
        mk.pose.position.x, mk.pose.position.y = self.qr_map[0], self.qr_map[1]
        mk.pose.position.z = 0.2
        mk.pose.orientation = yaw_to_quat(self.qr_map[2])
        mk.scale.x, mk.scale.y, mk.scale.z = 0.01, 0.1, 0.1
        mk.color = ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.8)
        self.pub_qr.publish(mk)

    def _maybe_debug(self, frame):
        if not self.publish_debug or frame is None:
            return
        now = self._now()
        if (now - self.last_debug_t) < self.debug_period:
            return
        if self.pub_dbg.get_subscription_count() == 0:
            return
        self.last_debug_t = now
        img = frame.copy()
        H, W = img.shape[:2]
        cv2.line(img, (W // 2, 0), (W // 2, H), (90, 90, 90), 1)
        if self.last_corners is not None:
            cv2.polylines(img, [self.last_corners.astype(int).reshape(-1, 1, 2)],
                          True, (0, 255, 0), 2)
        lines = [f"phase: {self.phase.value}",
                 f"observe: {len(self.obs)}/{self.observe_frames}"]
        if self.qr_map:
            lines.append(f"QR map: ({self.qr_map[0]:.2f},{self.qr_map[1]:.2f})")
        y = 18
        for ln in lines:
            cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out.header.stamp = self.get_clock().now().to_msg()
            self.pub_dbg.publish(out)
        except Exception:
            pass


def main():
    rclpy.init()
    node = QRDockMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
