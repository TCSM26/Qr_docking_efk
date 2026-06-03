#!/usr/bin/env python3
"""
qr_dock_kf_node.py
==================
Kalman-filtered QR docking for the TCSM Puzzlebot forklift.

Self-contained alternative to robot_dock_twostep. It does NOT touch the
EKF / MCL / ArUco localization stack: the global stack brings the robot
*near* the dock, this node performs the final precision docking onto a
fixed QR code.

Pipeline (inspired by dawan0111/Auto-Marker-Docking, adapted to QR):

  1. Measurement -- cv2.QRCodeDetector + cv2.solvePnPGeneric(IPPE) returns the
     two planar-PnP solutions; we disambiguate by reprojection error during
     warm-up and then by temporal consistency of the outward normal. The
     measurement is the QR pose in base_link: (mx, my, normal_yaw).

  2. Tracking -- a 3-DOF (x, y, yaw) Kalman filter whose PREDICT step is driven
     by the robot's own ego-motion (commanded or odom twist). Two consequences:
       * the marker yaw is propagated by ego-motion between vision updates, so a
         large measurement covariance on yaw (R_yaw >> R_xy) lets us *distrust*
         the noisy instantaneous plane-yaw -- the single most important trick
         for docking to a QR rotated about its vertical (Z) axis;
       * when the QR is briefly occluded or leaves the FOV at close range, the
         filter keeps a valid estimate (marker-loss prediction) instead of
         going blind.

  3. Docking -- a closed-loop state machine that routes the robot through a
     WAYPOINT placed on the QR's outward normal (this is what enforces a
     perpendicular approach WITHOUT trusting the instantaneous yaw), then
     re-references the QR itself for the final centering and standoff.

I/O (matches the other aligners so it is a drop-in on the cmd_vel_mux):
  subscribes : image_topic (sensor_msgs/Image)
               odom_topic  (nav_msgs/Odometry, optional, for ego-motion)
               /align/mode (std_msgs/String) -- activate on activate_mode
  publishes  : alignment_cmd_vel (geometry_msgs/Twist)
               /align/done       (std_msgs/Bool)
               ~/debug_image, ~/qr_marker, ~/waypoint  (debug / RViz)
  service    : enable_service (custom_interfaces/SetProcessBool)
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

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from std_msgs.msg import Bool, String, ColorRGBA
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

try:
    from custom_interfaces.srv import SetProcessBool
except ImportError:
    SetProcessBool = None


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quat(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def build_cam_extrinsics(x, y, z, pitch_rad):
    """base_link <- camera optical, same convention as qr_dock_twostep."""
    s = math.sin(pitch_rad)
    c = math.cos(pitch_rad)
    R = np.array([
        [0.0, -s,  c],
        [-1.0, 0.0, 0.0],
        [0.0, -c, -s],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


class Step(str, Enum):
    IDLE = "idle"
    DETECT = "detect"            # accumulate detections, hold still
    ALIGN_BEARING = "align_bearing"   # rotate to point at the normal-waypoint
    DRIVE = "drive"              # drive forward to the waypoint
    FINAL_ALIGN = "final_align"  # re-reference QR, rotate to center it
    DOCK = "dock"                # final approach to standoff (vision/KF)
    COMMIT = "commit"            # open-loop straight advance (QR too close to see)
    DONE = "done"


# --------------------------------------------------------------------------- #
#  3-DOF (x, y, yaw) Kalman filter with ego-motion prediction
# --------------------------------------------------------------------------- #
class MarkerKF:
    """Tracks the QR pose expressed in the *robot* (base_link) frame.

    State X = [mx, my, myaw]:
        mx, my  -- QR centre position in base_link [m]
        myaw    -- QR outward-normal direction in base_link [rad]

    predict(v, w, dt) transforms the (static-in-world) marker into the new
    base_link frame after the robot moves with linear v and angular w. This is
    an EKF predict (nonlinear mean, Jacobian-propagated covariance).
    """

    def __init__(self, P0, Q, R):
        self.X = None                 # not initialized until first measurement
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

    def initialized(self):
        return self.X is not None

    def init_state(self, z):
        self.X = np.array(z, dtype=float)

    def predict(self, v, w, dt):
        if self.X is None:
            return
        a = w * dt
        vd = v * dt
        ca, sa = math.cos(a), math.sin(a)
        mx, my, myaw = self.X
        # Same fixed point seen from the new base_link frame:
        #   p_{t+1} = R(-a) * (p_t - (vd, 0))
        nx = ca * (mx - vd) + sa * my
        ny = -sa * (mx - vd) + ca * my
        nyaw = wrap_to_pi(myaw - a)
        self.X = np.array([nx, ny, nyaw])
        J = np.array([[ca,  sa, 0.0],
                      [-sa, ca, 0.0],
                      [0.0, 0.0, 1.0]])
        self.P = J @ self.P @ J.T + self.Q

    def update(self, z):
        if self.X is None:
            self.init_state(z)
            return
        y = np.array(z, dtype=float) - self.X
        y[2] = wrap_to_pi(y[2])          # angle-aware innovation
        S = self.P + self.R              # H = I
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.X[2] = wrap_to_pi(self.X[2])
        self.P = (np.eye(3) - K) @ self.P


# --------------------------------------------------------------------------- #
#  Node
# --------------------------------------------------------------------------- #
class QRDockKFNode(Node):
    def __init__(self):
        super().__init__("qr_dock_kf_node")

        # ---- I/O wiring ------------------------------------------------------
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("cmd_vel_topic", "alignment_cmd_vel")
        self.declare_parameter("done_topic", "/align/done")
        self.declare_parameter("mode_topic", "/align/mode")
        self.declare_parameter("activate_mode", "dock_qr_kf")
        self.declare_parameter("enable_service", "/qr_dock_kf/enable")
        self.declare_parameter("use_odom_twist", True)

        # ---- camera / calibration -------------------------------------------
        self.declare_parameter("calib_path", "")
        self.declare_parameter("default_qr_size_mm", 97.0)
        self.declare_parameter("qr_size_overrides", "")
        self.declare_parameter("cam_x_offset_m", 0.14)
        self.declare_parameter("cam_y_offset_m", 0.0)
        self.declare_parameter("cam_z_offset_m", 0.205)
        self.declare_parameter("cam_pitch_deg", 0.0)

        # ---- measurement gating ---------------------------------------------
        self.declare_parameter("warmup_frames", 8)
        # Don't leave DETECT until the measured normal yaw is stable over the
        # recent window -- prevents locking onto a flipped planar-PnP branch.
        self.declare_parameter("lock_max_yaw_std_rad", 0.12)
        self.declare_parameter("max_reproj_px", 6.0)
        self.declare_parameter("max_jump_m", 0.30)
        self.declare_parameter("max_jump_rad", 1.2)
        # After warm-up, reject a measurement whose yaw disagrees with the KF
        # estimate by more than this -- a static QR + slow robot should never
        # jump, so a big disagreement is almost always the wrong PnP branch.
        self.declare_parameter("max_yaw_innov_rad", 0.6)
        self.declare_parameter("loss_timeout_s", 5.0)
        self.declare_parameter("debug", False)

        # ---- Kalman filter (note: large R on yaw = distrust instantaneous yaw)
        self.declare_parameter("init_cov", [0.05, 0.05, 0.20])
        self.declare_parameter("predict_cov", [0.005, 0.005, 0.002])
        self.declare_parameter("measure_cov", [0.004, 0.004, 0.20])

        # ---- docking geometry / control -------------------------------------
        self.declare_parameter("waypoint_dist_m", 0.25)   # standoff on the normal
        self.declare_parameter("marker_gap_m", 0.18)      # final stop distance
        self.declare_parameter("tol_lat_m", 0.02)         # lateral / bearing tol
        self.declare_parameter("tol_fwd_m", 0.03)         # forward tol at waypoint
        self.declare_parameter("stable_time_s", 0.4)

        # ---- open-loop commit (final advance once the QR is too close to see)
        # At close range the QR overflows the FOV and stops being detectable.
        # Rather than coast blindly on the KF until loss_timeout, we latch the
        # remaining distance and drive it straight by odometry, like the
        # dock_commit phase in qr_pose_align.
        self.declare_parameter("commit_enable", True)
        self.declare_parameter("commit_loss_s", 0.8)        # vision age that arms commit
        self.declare_parameter("commit_trigger_dist_m", 0.35)  # only commit when this close
        self.declare_parameter("commit_lat_tol_m", 0.06)    # require alignment to commit
        self.declare_parameter("commit_speed", 0.04)
        self.declare_parameter("commit_max_dist_m", 0.40)   # safety clamp on advance

        self.declare_parameter("kp_lin", 0.6)
        self.declare_parameter("kd_lin", 0.02)
        self.declare_parameter("kp_ang", 1.2)
        self.declare_parameter("kd_ang", 0.02)
        self.declare_parameter("max_lin_speed", 0.08)
        self.declare_parameter("max_ang_speed", 0.30)
        self.declare_parameter("min_lin_speed", 0.0)
        self.declare_parameter("min_ang_speed", 0.02)
        # Creep floors: minimum command toward an unsatisfied error so the
        # proportional controller doesn't stall asymptotically short of its
        # tolerance band.
        self.declare_parameter("creep_lin", 0.035)
        self.declare_parameter("creep_ang", 0.08)
        self.declare_parameter("approach_angle_gate_rad", 0.30)

        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_rate_hz", 10.0)
        self.declare_parameter("start_enabled", False)

        g = lambda n: self.get_parameter(n).value

        self.image_topic = g("image_topic")
        self.odom_topic = g("odom_topic")
        self.cmd_vel_topic = g("cmd_vel_topic")
        self.done_topic = g("done_topic")
        self.mode_topic = g("mode_topic")
        self.activate_mode = g("activate_mode")
        self.enable_service = g("enable_service")
        self.use_odom_twist = bool(g("use_odom_twist"))

        self.calib_path = g("calib_path")
        self.default_qr_size_m = g("default_qr_size_mm") / 1000.0
        self.size_overrides = self._parse_overrides(g("qr_size_overrides"))
        self.T_base_cam = build_cam_extrinsics(
            g("cam_x_offset_m"), g("cam_y_offset_m"), g("cam_z_offset_m"),
            math.radians(g("cam_pitch_deg")))

        self.warmup_frames = int(g("warmup_frames"))
        self.lock_max_yaw_std = float(g("lock_max_yaw_std_rad"))
        self.max_reproj_px = float(g("max_reproj_px"))
        self.max_jump_m = float(g("max_jump_m"))
        self.max_jump_rad = float(g("max_jump_rad"))
        self.max_yaw_innov = float(g("max_yaw_innov_rad"))
        self.loss_timeout = float(g("loss_timeout_s"))
        self.debug = bool(g("debug"))

        P0 = np.diag([float(x) for x in g("init_cov")])
        Q = np.diag([float(x) for x in g("predict_cov")])
        R = np.diag([float(x) for x in g("measure_cov")])
        self.kf = MarkerKF(P0, Q, R)

        self.d_wp = float(g("waypoint_dist_m"))
        self.marker_gap = float(g("marker_gap_m"))
        self.tol_lat = float(g("tol_lat_m"))
        self.tol_fwd = float(g("tol_fwd_m"))
        self.stable_time = float(g("stable_time_s"))

        self.kp_lin = float(g("kp_lin"))
        self.kd_lin = float(g("kd_lin"))
        self.kp_ang = float(g("kp_ang"))
        self.kd_ang = float(g("kd_ang"))
        self.v_max = float(g("max_lin_speed"))
        self.w_max = float(g("max_ang_speed"))
        self.v_min = float(g("min_lin_speed"))
        self.w_min = float(g("min_ang_speed"))
        self.creep_lin = float(g("creep_lin"))
        self.creep_ang = float(g("creep_ang"))
        self.approach_gate = float(g("approach_angle_gate_rad"))

        self.commit_enable = bool(g("commit_enable"))
        self.commit_loss_s = float(g("commit_loss_s"))
        self.commit_trigger_dist = float(g("commit_trigger_dist_m"))
        self.commit_lat_tol = float(g("commit_lat_tol_m"))
        self.commit_speed = float(g("commit_speed"))
        self.commit_max_dist = float(g("commit_max_dist_m"))

        self.rate_hz = float(g("control_rate_hz"))
        self.publish_debug = bool(g("publish_debug_image"))
        self.debug_period = 1.0 / max(0.1, g("debug_image_rate_hz"))

        (self.K, self.dist, self.cam_model) = self._load_calibration(self.calib_path)

        # ---- runtime state ---------------------------------------------------
        self.enabled = bool(g("start_enabled"))
        self.step = Step.IDLE
        self.detect_count = 0
        self.recent_yaws = []         # rolling window of measured normal yaws
        self.last_meas_t = None
        self.last_n_cam = None        # for PnP disambiguation
        self.last_t_cam = None
        self.cmd_v = 0.0              # last commanded velocity (ego-motion)
        self.cmd_w = 0.0
        self.odom_v = 0.0
        self.odom_w = 0.0
        self.odom_t = None
        self.odom_px = 0.0           # odom pose (for open-loop commit distance)
        self.odom_py = 0.0
        self.commit_start_x = 0.0
        self.commit_start_y = 0.0
        self.commit_target = 0.0
        self.prev_tick_t = None
        self.in_tol_since = None
        self.done_latched = False
        self.prev_err_lin = 0.0
        self.prev_err_ang = 0.0

        # cache for debug overlay
        self.last_corners = None
        self.last_payload = ""
        self.last_debug_pub_t = 0.0
        self.last_qr_pos_base = None

        self.bridge = CvBridge()
        self.qr_det = cv2.QRCodeDetector()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub_img = self.create_subscription(
            Image, self.image_topic, self._on_image, sensor_qos)
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, 10)
        self.sub_mode = self.create_subscription(
            String, self.mode_topic, self._on_mode, 10)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_done = self.create_publisher(Bool, self.done_topic, 10)
        self.pub_dbg = self.create_publisher(Image, "~/debug_image", 10)
        self.pub_marker = self.create_publisher(Marker, "~/qr_marker", 10)
        self.pub_wp = self.create_publisher(PoseStamped, "~/waypoint", 10)

        if SetProcessBool is not None:
            self.srv_enable = self.create_service(
                SetProcessBool, self.enable_service, self._on_enable)
        else:
            self.get_logger().warn(
                "custom_interfaces/SetProcessBool unavailable; enable service disabled.")

        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f"qr_dock_kf ready. activate via /align/mode='{self.activate_mode}' "
            f"or service {self.enable_service}. enabled={self.enabled}")

    # --------------------------------------------------------------------- #
    #  Parameter / calibration helpers
    # --------------------------------------------------------------------- #
    def _parse_overrides(self, s):
        out = {}
        if not s:
            return out
        for item in s.split(","):
            item = item.strip()
            if not item:
                continue
            idx = item.rfind("=")
            if idx == -1:
                continue
            key = item[:idx].strip()
            try:
                out[key] = float(item[idx + 1:].strip()) / 1000.0
            except ValueError:
                continue
        return out

    def _qr_size_for(self, content):
        if content:
            m = re.search(r"qr_mm\s*=\s*([0-9.]+)", content)
            if m:
                try:
                    return float(m.group(1)) / 1000.0
                except ValueError:
                    pass
            for key, val in self.size_overrides.items():
                if key in content:
                    return val
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
        K = np.array([[fx, 0, 160.0], [0, fy, 120.0], [0, 0, 1.0]])
        return K, np.zeros((5, 1)), "pinhole"

    # --------------------------------------------------------------------- #
    #  Activation
    # --------------------------------------------------------------------- #
    def _start(self):
        self.enabled = True
        self.step = Step.DETECT
        self.detect_count = 0
        self.kf.X = None
        self.last_meas_t = None
        self.last_n_cam = None
        self.last_t_cam = None
        self.in_tol_since = None
        self.done_latched = False
        self.prev_err_lin = 0.0
        self.prev_err_ang = 0.0
        self._publish_zero()
        self.get_logger().info("qr_dock_kf ENABLED -> DETECT")

    def _stop(self):
        self.enabled = False
        self.step = Step.IDLE
        self._publish_zero()

    def _on_enable(self, req, resp):
        if bool(req.enable):
            self._start()
        else:
            self._stop()
        resp.success = True
        resp.message = "enabled" if self.enabled else "disabled"
        return resp

    def _on_mode(self, msg):
        m = msg.data.strip()
        if m == self.activate_mode:
            if not self.enabled:
                self._start()
        else:
            if self.enabled:
                self._stop()

    def _on_odom(self, msg):
        self.odom_v = msg.twist.twist.linear.x
        self.odom_w = msg.twist.twist.angular.z
        self.odom_px = msg.pose.pose.position.x
        self.odom_py = msg.pose.pose.position.y
        self.odom_t = self._now()

    # --------------------------------------------------------------------- #
    #  Measurement: QR -> (mx, my, normal_yaw) in base_link
    # --------------------------------------------------------------------- #
    def _on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        data, points, _ = self.qr_det.detectAndDecode(frame)
        if points is None or len(points) == 0:
            self.last_corners = None
            self._maybe_publish_debug(frame)
            return
        corners = np.asarray(points).reshape(-1, 2).astype(np.float32)
        if corners.shape[0] != 4:
            self.last_corners = None
            self._maybe_publish_debug(frame)
            return
        self.last_corners = corners
        self.last_payload = data or self.last_payload

        size_m = self._qr_size_for(data)
        half = size_m / 2.0
        obj = np.array([
            [-half,  half, 0.0],   # TL  (QR-local: x right, y up, z out)
            [ half,  half, 0.0],   # TR
            [ half, -half, 0.0],   # BR
            [-half, -half, 0.0],   # BL
        ], dtype=np.float32)

        corners_pnp = corners.reshape(-1, 1, 2).astype(np.float64)
        if self.cam_model == "fisheye":
            corners_pnp = cv2.fisheye.undistortPoints(
                corners_pnp, self.K, self.dist, P=self.K)
            dist_pnp = np.zeros((4, 1), dtype=np.float64)
        else:
            dist_pnp = self.dist

        # IPPE returns the two planar-PnP solutions (the real one and its
        # near-mirror). Disambiguate -> the heart of robust QR pose.
        try:
            n_sols, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                obj, corners_pnp.reshape(-1, 2).astype(np.float32),
                self.K, dist_pnp, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            self._maybe_publish_debug(frame)
            return
        if not n_sols:
            self._maybe_publish_debug(frame)
            return

        cands = []
        for i in range(n_sols):
            Ri, _ = cv2.Rodrigues(rvecs[i])
            ti = tvecs[i].flatten()
            ni = Ri @ np.array([0.0, 0.0, 1.0])
            ei = float(reproj[i]) if reproj is not None else 0.0
            cands.append((Ri, ti, ni, ei))

        # A marker we can SEE must have its front face toward the camera, i.e.
        # its outward normal points back at the camera: n_cam_z < 0. This prior
        # resolves the planar-PnP two-fold ambiguity far more reliably than the
        # near-degenerate reprojection error of a fronto-parallel square.
        pool = [c for c in cands if c[2][2] < 0.0] or cands
        if self.last_n_cam is None:
            best_c = min(pool, key=lambda c: c[3])              # lowest reproj
        else:
            best_c = max(pool, key=lambda c: float(np.dot(c[2], self.last_n_cam)))

        R_cam, t_cam, n_cam, reproj_px = best_c
        if reproj_px > self.max_reproj_px:
            self._maybe_publish_debug(frame)
            return

        if self.debug and self.detect_count < self.warmup_frames + 2:
            Rbc = self.T_base_cam[:3, :3]
            info = "  ".join(
                f"#{i}(z={c[2][2]:+.2f},reproj={c[3]:.2f},"
                f"yaw_base={math.degrees(math.atan2(*(Rbc @ c[2])[[1, 0]])):+.0f})"
                for i, c in enumerate(cands))
            chosen = math.degrees(math.atan2(*(Rbc @ n_cam)[[1, 0]]))
            self.get_logger().info(f"PnP cands: {info} | chosen_yaw_base={chosen:+.0f}")

        # Reject implausible jumps (KF handles yaw noise, but a teleport is bad).
        if self.last_t_cam is not None:
            if np.linalg.norm(t_cam - self.last_t_cam) > self.max_jump_m:
                self._maybe_publish_debug(frame)
                return
            dn = float(np.arccos(np.clip(np.dot(n_cam, self.last_n_cam), -1.0, 1.0)))
            if dn > self.max_jump_rad:
                self._maybe_publish_debug(frame)
                return

        self.last_n_cam = n_cam
        self.last_t_cam = t_cam

        # Into base_link.
        qr_pos_base = (self.T_base_cam @ np.append(t_cam, 1.0))[:3]
        n_base = self.T_base_cam[:3, :3] @ n_cam
        normal_yaw = math.atan2(n_base[1], n_base[0])

        # Yaw-flip guard: once tracking, a measurement whose yaw disagrees
        # wildly with the KF is almost certainly the wrong planar-PnP branch.
        if self.kf.initialized() and self.detect_count >= self.warmup_frames:
            if abs(wrap_to_pi(normal_yaw - self.kf.X[2])) > self.max_yaw_innov:
                self._maybe_publish_debug(frame)
                return

        z = (float(qr_pos_base[0]), float(qr_pos_base[1]), float(normal_yaw))
        self.kf.update(z)
        self.detect_count += 1
        self.recent_yaws.append(float(normal_yaw))
        if len(self.recent_yaws) > self.warmup_frames:
            self.recent_yaws.pop(0)
        self.last_meas_t = self._now()
        self.last_qr_pos_base = qr_pos_base

        if data:
            self.last_payload = data
        self._publish_marker(qr_pos_base, normal_yaw, size_m)
        self._maybe_publish_debug(frame)

    # --------------------------------------------------------------------- #
    #  Control loop / state machine
    # --------------------------------------------------------------------- #
    def _control_loop(self):
        now = self._now()
        dt = (now - self.prev_tick_t) if self.prev_tick_t else (1.0 / self.rate_hz)
        self.prev_tick_t = now
        dt = max(1e-3, min(0.2, dt))

        if not self.enabled or self.step in (Step.IDLE, Step.DONE):
            if self.step == Step.DONE:
                self._publish_zero()
            return

        # Ego-motion source for the KF predict.
        if self.use_odom_twist and self.odom_t is not None \
                and (now - self.odom_t) < 0.5:
            v_eg, w_eg = self.odom_v, self.odom_w
        else:
            v_eg, w_eg = self.cmd_v, self.cmd_w
        self.kf.predict(v_eg, w_eg, dt)

        # Open-loop commit is intentionally blind -> handle it before any of the
        # vision-loss / abort logic below.
        if self.step == Step.COMMIT:
            self._tick_commit()
            return

        # No estimate yet -> wait.
        if not self.kf.initialized():
            self._publish_zero()
            return

        vision_age = float('inf') if self.last_meas_t is None \
            else (now - self.last_meas_t)

        # DETECT needs live vision to lock on; it never coasts.
        if self.step == Step.DETECT and vision_age > 0.5:
            self._publish_zero()
            if vision_age > self.loss_timeout:
                self.get_logger().warn("QR lost during DETECT; aborting.")
                self.step = Step.IDLE
                self.enabled = False
            return

        # Driving stages COAST on the KF ego-motion prediction (marker-loss
        # prediction) when the QR drops out of the FOV -- e.g. it swings out of
        # view on an oblique approach, then re-enters once we turn to face it.
        # Abort only if it stays unseen past loss_timeout.
        if vision_age > self.loss_timeout:
            self.get_logger().warn(
                f"QR unseen > {self.loss_timeout:.1f}s in {self.step.value}; aborting.",
                throttle_duration_sec=1.0)
            self._publish_zero()
            self.step = Step.IDLE
            self.enabled = False
            return
        if vision_age > 0.3:
            self.get_logger().info("coasting on KF (QR not visible)",
                                   throttle_duration_sec=1.0)

        mx, my, myaw = self.kf.X
        # Waypoint on the QR outward normal, d_wp in front of the marker.
        wx = mx + self.d_wp * math.cos(myaw)
        wy = my + self.d_wp * math.sin(myaw)
        self._publish_waypoint(wx, wy, myaw)

        if self.debug:
            self.get_logger().info(
                f"[{self.step.value}] mx={mx:.2f} my={my:.2f} myaw={math.degrees(myaw):.0f} "
                f"wx={wx:.2f} wy={wy:.2f} bearing_wp={math.degrees(math.atan2(wy,wx)):.0f} "
                f"age={vision_age:.1f}", throttle_duration_sec=0.4)

        if self.step == Step.DETECT:
            self._publish_zero()
            # Lock only when we have enough frames AND the measured normal yaw
            # is stable (low circular std) -- so we never start driving on a
            # flipped / coin-flip planar-PnP branch.
            if self.detect_count >= self.warmup_frames \
                    and len(self.recent_yaws) >= self.warmup_frames:
                s = np.mean(np.sin(self.recent_yaws))
                c = np.mean(np.cos(self.recent_yaws))
                yaw_std = math.sqrt(-2.0 * math.log(max(1e-9, min(1.0, math.hypot(s, c)))))
                if yaw_std <= self.lock_max_yaw_std:
                    self.step = Step.ALIGN_BEARING
                    self.get_logger().info(
                        f"DETECT -> ALIGN_BEARING (yaw_std={math.degrees(yaw_std):.1f}deg)")
                else:
                    self.get_logger().warn(
                        f"normal unstable (yaw_std={math.degrees(yaw_std):.1f}deg); waiting",
                        throttle_duration_sec=1.0)
            return

        if self.step == Step.ALIGN_BEARING:
            # Rotate in place until the waypoint is straight ahead.
            bearing = math.atan2(wy, wx)
            w = self._ang_cmd(bearing, dt)
            self._publish_cmd(0.0, w)
            if abs(wy) <= self.tol_lat or abs(bearing) <= self.tol_lat:
                self.step = Step.DRIVE
                self.get_logger().info("ALIGN_BEARING -> DRIVE")
            return

        if self.step == Step.DRIVE:
            # Drive to the waypoint (point-and-go: forward + light steering).
            bearing = math.atan2(wy, wx)
            w = self._ang_ctrl(bearing, dt)
            v = self._lin_cmd(wx, dt) if abs(bearing) < self.approach_gate else 0.0
            self._publish_cmd(v, w)
            if abs(wx) <= self.tol_fwd:
                self.step = Step.FINAL_ALIGN
                self.get_logger().info("DRIVE -> FINAL_ALIGN")
            return

        if self.step == Step.FINAL_ALIGN:
            # Re-reference the QR itself: rotate to center it laterally.
            bearing = math.atan2(my, mx)
            w = self._ang_cmd(bearing, dt)
            self._publish_cmd(0.0, w)
            if abs(my) <= self.tol_lat or abs(bearing) <= self.tol_lat:
                self.step = Step.DOCK
                self.get_logger().info("FINAL_ALIGN -> DOCK")
            return

        if self.step == Step.DOCK:
            # Arm the open-loop commit once we're close and the QR has dropped
            # out of view (it overflows the FOV at short range and stops being
            # detectable). Latch the remaining distance and drive it by odom.
            if self.commit_enable and vision_age > self.commit_loss_s \
                    and mx <= self.commit_trigger_dist \
                    and abs(my) <= self.commit_lat_tol:
                self.commit_start_x = self.odom_px
                self.commit_start_y = self.odom_py
                self.commit_target = float(
                    np.clip(mx - self.marker_gap, 0.0, self.commit_max_dist))
                self.step = Step.COMMIT
                self.get_logger().info(
                    f"DOCK -> COMMIT: advance {self.commit_target:.3f} m open-loop (odom)")
                return

            bearing = math.atan2(my, mx)
            fwd_err = mx - self.marker_gap
            ang_done = abs(my) <= self.tol_lat
            lin_done = fwd_err <= self.tol_fwd
            w = 0.0 if ang_done else self._ang_cmd(bearing, dt)
            v = 0.0 if (lin_done or abs(bearing) >= self.approach_gate) \
                else self._lin_cmd(fwd_err, dt)
            self._publish_cmd(v, w)

            if lin_done and ang_done:
                if self.in_tol_since is None:
                    self.in_tol_since = now
                elif (now - self.in_tol_since) >= self.stable_time:
                    self._finish()
            else:
                self.in_tol_since = None
            return

    def _tick_commit(self):
        """Open-loop final advance: drive straight by odometry the latched
        remaining distance, then declare success. Used when the QR is too close
        to detect, so neither vision nor KF coasting can close the last gap."""
        traveled = math.hypot(self.odom_px - self.commit_start_x,
                              self.odom_py - self.commit_start_y)
        if traveled >= self.commit_target:
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info(f"COMMIT done ({traveled:.3f} m advanced).")
            self._finish()
            return
        self._publish_cmd(self.commit_speed, 0.0)

    def _finish(self):
        self.step = Step.DONE
        self._publish_zero()
        if not self.done_latched:
            self.pub_done.publish(Bool(data=True))
            self.done_latched = True
            self.get_logger().info("DOCK complete; /align/done=true.")

    # --------------------------------------------------------------------- #
    #  Controllers (PD on scalar errors)
    # --------------------------------------------------------------------- #
    def _lin_ctrl(self, err, dt):
        d = (err - self.prev_err_lin) / dt
        self.prev_err_lin = err
        return self.kp_lin * err + self.kd_lin * d

    def _ang_ctrl(self, err, dt):
        d = (err - self.prev_err_ang) / dt
        self.prev_err_ang = err
        return self.kp_ang * err + self.kd_ang * d

    def _lin_cmd(self, err, dt):
        """Linear PD with a creep floor so it reaches the tolerance band."""
        v = self._lin_ctrl(err, dt)
        if abs(v) < self.creep_lin:
            v = math.copysign(self.creep_lin, err)
        return v

    def _ang_cmd(self, err, dt):
        """Angular PD with a creep floor toward the unsatisfied heading error."""
        w = self._ang_ctrl(err, dt)
        if abs(w) < self.creep_ang:
            w = math.copysign(self.creep_ang, err)
        return w

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
        self.cmd_v, self.cmd_w = v, w
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub_cmd.publish(m)

    def _publish_zero(self):
        self.cmd_v, self.cmd_w = 0.0, 0.0
        self.pub_cmd.publish(Twist())

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # --------------------------------------------------------------------- #
    #  Visualization
    # --------------------------------------------------------------------- #
    def _publish_waypoint(self, wx, wy, myaw):
        if self.pub_wp.get_subscription_count() == 0:
            return
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "base_footprint"
        ps.pose.position.x = float(wx)
        ps.pose.position.y = float(wy)
        ps.pose.orientation = yaw_to_quat(wrap_to_pi(myaw + math.pi))
        self.pub_wp.publish(ps)

    def _publish_marker(self, pos, normal_yaw, size_m):
        if self.pub_marker.get_subscription_count() == 0:
            return
        mk = Marker()
        mk.header.frame_id = "base_footprint"
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "qr_dock_kf"
        mk.id = 0
        mk.type = Marker.CUBE
        mk.action = Marker.ADD
        mk.pose.position.x = float(pos[0])
        mk.pose.position.y = float(pos[1])
        mk.pose.position.z = float(pos[2])
        mk.pose.orientation = yaw_to_quat(normal_yaw)
        mk.scale.x = 0.01
        mk.scale.y = float(size_m)
        mk.scale.z = float(size_m)
        mk.color = ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.8)
        mk.lifetime.nanosec = int(3e8)
        self.pub_marker.publish(mk)

    def _maybe_publish_debug(self, frame):
        if not self.publish_debug or frame is None:
            return
        now = self._now()
        if (now - self.last_debug_pub_t) < self.debug_period:
            return
        if self.pub_dbg.get_subscription_count() == 0:
            return
        self.last_debug_pub_t = now

        img = frame.copy()
        H, W = img.shape[:2]
        cv2.line(img, (W // 2, 0), (W // 2, H), (90, 90, 90), 1)
        if self.last_corners is not None:
            pts = self.last_corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cx = int(self.last_corners[:, 0].mean())
            cy = int(self.last_corners[:, 1].mean())
            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)

        lines = [f"step: {self.step.value}  enabled: {self.enabled}",
                 f"detections: {self.detect_count}/{self.warmup_frames}"]
        if self.kf.initialized():
            mx, my, myaw = self.kf.X
            lines.append(f"QR base: x={mx:+.2f} y={my:+.2f} yaw={math.degrees(myaw):+.0f}deg")
        if self.last_payload:
            lines.append(f"payload: {self.last_payload[:28]}")
        y = 18
        for ln in lines:
            cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = "camera_frame"
            self.pub_dbg.publish(out)
        except Exception:
            pass


def main():
    rclpy.init()
    node = QRDockKFNode()
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
