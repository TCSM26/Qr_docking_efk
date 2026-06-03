#!/usr/bin/env python3
"""
qr_sim_world.py
===============
Lightweight synthetic world to test qr_dock_kf_node WITHOUT Gazebo.

It closes the loop entirely in software:

  * holds a fixed QR on a (virtual) wall at a configurable pose, including a
    Z-axis rotation (the hard case we want to validate);
  * renders what a pinhole camera mounted on the robot would see and publishes
    it as sensor_msgs/Image on `image_topic`;
  * subscribes to the robot's `cmd_vel_topic`, integrates a unicycle model, and
    republishes the resulting twist on `odom_topic` (so the docking node's
    ego-motion prediction is fed the true velocity);
  * prints success metrics (lateral offset from the QR normal, standoff
    distance and heading error) when `/align/done` fires.

Camera intrinsics MATCH qr_dock_kf_node's built-in fallback (320x240, ~60deg
HFOV), so run the docking node with an empty calib_path and the same QR size.

Run via the combined launch, or standalone:
    ros2 run qr_dock_kf qr_sim_world
"""

import math
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def rmat_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def build_cam_extrinsics(x, y, z, pitch_rad):
    """Identical to qr_dock_kf_node: maps camera-optical coords -> base_link."""
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


def T_world_base(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([
        [c, -s, 0.0, x],
        [s,  c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


class QRSimWorld(Node):
    def __init__(self):
        super().__init__("qr_sim_world")

        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("cmd_vel_topic", "alignment_cmd_vel")
        self.declare_parameter("done_topic", "/align/done")

        # Image / camera. If calib_path is set, K is loaded from it so the sim
        # and the docking node use byte-identical intrinsics; otherwise the
        # intrinsics are derived from img_w/img_h/hfov_deg.
        self.declare_parameter("calib_path", "")
        self.declare_parameter("img_w", 320)
        self.declare_parameter("img_h", 240)
        self.declare_parameter("hfov_deg", 60.0)
        self.declare_parameter("cam_x_offset_m", 0.14)
        self.declare_parameter("cam_y_offset_m", 0.0)
        self.declare_parameter("cam_z_offset_m", 0.205)
        self.declare_parameter("cam_pitch_deg", 0.0)
        self.declare_parameter("fps", 30.0)

        # Robot start pose (world)
        self.declare_parameter("robot_x", 0.0)
        self.declare_parameter("robot_y", 0.0)
        self.declare_parameter("robot_theta", 0.0)

        # QR pose (world). normal_yaw is the outward normal direction; the
        # z_tilt_deg is added to a "facing the robot" baseline so positive
        # values rotate the QR about its vertical axis.
        self.declare_parameter("qr_x", 0.75)
        self.declare_parameter("qr_y", 0.10)
        self.declare_parameter("qr_z", 0.205)
        self.declare_parameter("qr_size_m", 0.20)
        self.declare_parameter("qr_z_tilt_deg", 25.0)
        self.declare_parameter("qr_payload", "qr_mm=200")

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g("image_topic")
        self.odom_topic = g("odom_topic")
        self.cmd_vel_topic = g("cmd_vel_topic")

        self.W = int(g("img_w"))
        self.H = int(g("img_h"))
        calib = g("calib_path")
        if calib and os.path.isfile(calib):
            data = np.load(calib, allow_pickle=True)
            K = (data["K"] if "K" in data.files else data["mtx"]).astype(float)
            self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
            self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
            self.get_logger().info(f"loaded calib from {calib}: fx={self.fx:.1f}")
        else:
            hfov = math.radians(g("hfov_deg"))
            self.fx = (self.W / 2.0) / math.tan(hfov / 2.0)
            self.fy = self.fx
            self.cx = self.W / 2.0
            self.cy = self.H / 2.0
        self.T_base_cam = build_cam_extrinsics(
            g("cam_x_offset_m"), g("cam_y_offset_m"), g("cam_z_offset_m"),
            math.radians(g("cam_pitch_deg")))

        self.rx = float(g("robot_x"))
        self.ry = float(g("robot_y"))
        self.rth = float(g("robot_theta"))

        self.qr_x = float(g("qr_x"))
        self.qr_y = float(g("qr_y"))
        self.qr_z = float(g("qr_z"))
        self.qr_size = float(g("qr_size_m"))
        # Baseline: outward normal points from the QR back toward the robot's
        # start; add the tilt so the QR is rotated about Z.
        face = math.atan2(self.ry - self.qr_y, self.rx - self.qr_x)
        self.qr_normal_yaw = face + math.radians(g("qr_z_tilt_deg"))
        self.get_logger().info(
            f"QR at ({self.qr_x:.2f},{self.qr_y:.2f}) normal_yaw="
            f"{math.degrees(self.qr_normal_yaw):.1f}deg, tilt={g('qr_z_tilt_deg')}deg")

        self.qr_corners_world = self._qr_world_corners()
        self.qr_tex, self.qr_src = self._make_qr_texture(g("qr_payload"))

        self.v = 0.0
        self.w = 0.0
        self.last_cmd_t = None
        self.bridge = CvBridge()
        self.done = False
        self.t0 = self._now()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_img = self.create_publisher(Image, self.image_topic, sensor_qos)
        # Also offer a compressed stream (matches a real camera) for testing the
        # node's use_compressed_image path.
        self.pub_img_c = self.create_publisher(
            CompressedImage, self.image_topic + "/compressed", sensor_qos)
        self.pub_odom = self.create_publisher(Odometry, self.odom_topic, 10)
        # Emulate the on-Jetson QR node: camera->QR pose (+X out of face) and
        # payload, so the docking node's external-pose path can be tested.
        self.declare_parameter("publish_qr_pose", True)
        from rcl_interfaces.msg import ParameterDescriptor
        self.declare_parameter("qr_id", "1",
                               ParameterDescriptor(dynamic_typing=True))
        self.publish_qr_pose = bool(g("publish_qr_pose"))
        self.qr_id = str(g("qr_id"))
        self.pub_qrpose = self.create_publisher(PoseStamped, "/qr/pose", 10)
        self.pub_qrdata = self.create_publisher(String, "/qr/data", 10)
        self.sub_cmd = self.create_subscription(
            Twist, self.cmd_vel_topic, self._on_cmd, 10)
        self.sub_done = self.create_subscription(
            Bool, g("done_topic"), self._on_done, 10)

        # TF: publish a (ground-truth) localization tree so the map-frame
        # docking node can look up map->base_footprint. map->odom is identity
        # (perfect localization); odom->base_footprint carries the true pose.
        self.tf_bcast = TransformBroadcaster(self)
        self.static_bcast = StaticTransformBroadcaster(self)
        # Optional localization error on map->odom (emulating EKF/MCL).
        #  - a CONSTANT bias is harmless here (the QR is localized in the same
        #    biased frame the robot navigates in -> it cancels), which is a nice
        #    property; the meaningful stressors are drift and jitter DURING the
        #    maneuver, which the REFINE visual correction must absorb.
        self.declare_parameter("loc_bias_x", 0.0)
        self.declare_parameter("loc_bias_y", 0.0)
        self.declare_parameter("loc_bias_yaw_deg", 0.0)
        self.declare_parameter("loc_drift_x", 0.0)       # m/s
        self.declare_parameter("loc_drift_y", 0.0)       # m/s
        self.declare_parameter("loc_jitter_m", 0.0)      # per-tick stddev
        self.loc_bias = (float(g("loc_bias_x")), float(g("loc_bias_y")),
                         math.radians(float(g("loc_bias_yaw_deg"))))
        self.loc_drift = (float(g("loc_drift_x")), float(g("loc_drift_y")))
        self.loc_jitter = float(g("loc_jitter_m"))

        # Static base_footprint -> camera (camera-optical pose in base, = T_base_cam),
        # so the external /qr/pose (stamped in 'camera') is TF-resolvable.
        b2c = TransformStamped()
        b2c.header.stamp = self.get_clock().now().to_msg()
        b2c.header.frame_id = "base_footprint"
        b2c.child_frame_id = "camera"
        b2c.transform.translation.x = float(self.T_base_cam[0, 3])
        b2c.transform.translation.y = float(self.T_base_cam[1, 3])
        b2c.transform.translation.z = float(self.T_base_cam[2, 3])
        cqx, cqy, cqz, cqw = rmat_to_quat(self.T_base_cam[:3, :3])
        b2c.transform.rotation.x = cqx
        b2c.transform.rotation.y = cqy
        b2c.transform.rotation.z = cqz
        b2c.transform.rotation.w = cqw
        self.static_bcast.sendTransform(b2c)

        dt = 1.0 / float(g("fps"))
        self.dt = dt
        self.timer = self.create_timer(dt, self._tick)
        self.last_log = 0.0

    # ------------------------------------------------------------------ #
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _qr_world_corners(self):
        nrm = self.qr_normal_yaw
        r = np.array([-math.sin(nrm), math.cos(nrm), 0.0])  # QR-local +x (right)
        u = np.array([0.0, 0.0, 1.0])                        # QR-local +y (up)
        C = np.array([self.qr_x, self.qr_y, self.qr_z])
        h = self.qr_size / 2.0
        # order TL, TR, BR, BL (matches the node's obj points)
        return np.array([
            C - h * r + h * u,
            C + h * r + h * u,
            C + h * r - h * u,
            C - h * r - h * u,
        ])

    def _make_qr_texture(self, payload):
        enc = cv2.QRCodeEncoder_create()
        qr = enc.encode(payload)                       # uint8, 0/255 modules
        module_px = 480
        qr = cv2.resize(qr, (module_px, module_px), interpolation=cv2.INTER_NEAREST)
        b = 120                                        # extra white quiet zone
        tex = np.full((module_px + 2 * b, module_px + 2 * b), 255, np.uint8)
        tex[b:b + module_px, b:b + module_px] = qr
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        # The encoder embeds its own quiet zone, so the code modules the
        # detector returns are INSET from our drawn area. Self-calibrate `src`
        # by detecting the QR on the (fronto-parallel) texture: those are the
        # exact corners the detector will report, so the homography maps them
        # to the true physical world corners and the recovered scale is right.
        det = cv2.QRCodeDetector()
        _, pts, _ = det.detectAndDecode(tex)
        if pts is not None and len(pts) > 0:
            src = np.asarray(pts).reshape(-1, 2).astype(np.float32)
        else:
            src = np.array([
                [b, b], [b + module_px, b],
                [b + module_px, b + module_px], [b, b + module_px],
            ], dtype=np.float32)
            self.get_logger().warn("QR self-calibration failed; using nominal corners.")
        return tex, src

    def _project(self, p_world, Rwc, twc):
        p_cam = Rwc.T @ (p_world - twc)
        Z = p_cam[2]
        if Z <= 0.05:
            return None
        u = self.fx * p_cam[0] / Z + self.cx
        v = self.fy * p_cam[1] / Z + self.cy
        return np.array([u, v]), Z

    def _render(self):
        img = np.full((self.H, self.W, 3), 140, np.uint8)   # gray background
        T_wc = T_world_base(self.rx, self.ry, self.rth) @ self.T_base_cam
        Rwc = T_wc[:3, :3]
        twc = T_wc[:3, 3]

        dst = []
        for c in self.qr_corners_world:
            pr = self._project(c, Rwc, twc)
            if pr is None:
                return img  # a corner is behind the camera -> QR not visible
            dst.append(pr[0])
        dst = np.array(dst, dtype=np.float32)

        # Reject if mostly out of frame.
        m = 40
        if (dst[:, 0].max() < -m or dst[:, 0].min() > self.W + m or
                dst[:, 1].max() < -m or dst[:, 1].min() > self.H + m):
            return img

        Hm = cv2.getPerspectiveTransform(self.qr_src, dst)
        warped = cv2.warpPerspective(self.qr_tex, Hm, (self.W, self.H),
                                     flags=cv2.INTER_LINEAR, borderValue=(140, 140, 140))
        mask = cv2.warpPerspective(
            np.full(self.qr_tex.shape[:2], 255, np.uint8), Hm, (self.W, self.H))
        img[mask > 0] = warped[mask > 0]
        # mild sensor noise
        noise = np.random.normal(0, 2.0, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    # ------------------------------------------------------------------ #
    def _publish_qr_pose(self, stamp):
        T_wc = T_world_base(self.rx, self.ry, self.rth) @ self.T_base_cam
        Rwc, twc = T_wc[:3, :3], T_wc[:3, 3]
        C = np.array([self.qr_x, self.qr_y, self.qr_z])
        p_cam = Rwc.T @ (C - twc)
        Z = p_cam[2]
        if Z <= 0.1:
            return  # behind camera -> "not detected"
        u = self.fx * p_cam[0] / Z + self.cx
        v = self.fy * p_cam[1] / Z + self.cy
        if not (-20 <= u <= self.W + 20 and -20 <= v <= self.H + 20):
            return  # out of frame -> "not detected"
        # QR orientation in world (+X out of face, +Z up), then into camera.
        nrm = self.qr_normal_yaw
        Xax = np.array([math.cos(nrm), math.sin(nrm), 0.0])
        Zax = np.array([0.0, 0.0, 1.0])
        Yax = np.cross(Zax, Xax)
        R_world_qr = np.column_stack((Xax, Yax, Zax))
        R_cam_qr = Rwc.T @ R_world_qr
        qx, qy, qz, qw = rmat_to_quat(R_cam_qr)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = "camera"
        ps.pose.position.x = float(p_cam[0])
        ps.pose.position.y = float(p_cam[1])
        ps.pose.position.z = float(p_cam[2])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pub_qrpose.publish(ps)
        self.pub_qrdata.publish(String(data='{"id": "%s"}' % self.qr_id))

    def _on_cmd(self, msg):
        self.v = float(msg.linear.x)
        self.w = float(msg.angular.z)
        self.last_cmd_t = self._now()

    def _on_done(self, msg):
        if msg.data and not self.done:
            self.done = True
            self._report()

    def _report(self):
        nrm = self.qr_normal_yaw
        # vector from QR center to robot
        dx = self.rx - self.qr_x
        dy = self.ry - self.qr_y
        dist_to_qr = math.hypot(dx, dy)
        # lateral offset from the outward-normal axis
        nx, ny = math.cos(nrm), math.sin(nrm)
        lateral = abs(-ny * dx + nx * dy)
        # heading error: robot should face the QR center
        desired_heading = math.atan2(self.qr_y - self.ry, self.qr_x - self.rx)
        heading_err = math.degrees(abs(math.atan2(
            math.sin(self.rth - desired_heading),
            math.cos(self.rth - desired_heading))))
        self.get_logger().info(
            "===== DOCK DONE ====="
            f"\n  time           : {self._now() - self.t0:.1f} s"
            f"\n  standoff dist  : {dist_to_qr:.3f} m"
            f"\n  lateral offset : {lateral*100:.1f} cm (from QR normal axis)"
            f"\n  heading error  : {heading_err:.1f} deg"
            f"\n  robot pose     : x={self.rx:.3f} y={self.ry:.3f} th={math.degrees(self.rth):.1f}deg")

    def _tick(self):
        now = self._now()
        # decay command if stale (node stopped publishing)
        if self.last_cmd_t is not None and (now - self.last_cmd_t) > 0.3:
            self.v = 0.0
            self.w = 0.0

        # integrate unicycle
        self.rx += self.v * math.cos(self.rth) * self.dt
        self.ry += self.v * math.sin(self.rth) * self.dt
        self.rth = math.atan2(math.sin(self.rth + self.w * self.dt),
                              math.cos(self.rth + self.w * self.dt))

        # publish odom (twist is what the docking node uses for ego-motion)
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = "odom"
        od.child_frame_id = "base_footprint"
        od.pose.pose.position.x = self.rx
        od.pose.pose.position.y = self.ry
        od.pose.pose.orientation.z = math.sin(self.rth / 2.0)
        od.pose.pose.orientation.w = math.cos(self.rth / 2.0)
        od.twist.twist.linear.x = self.v
        od.twist.twist.angular.z = self.w
        self.pub_odom.publish(od)

        # map -> odom (the believed-localization error: bias + drift + jitter)
        t = now - self.t0
        ex = self.loc_bias[0] + self.loc_drift[0] * t
        ey = self.loc_bias[1] + self.loc_drift[1] * t
        if self.loc_jitter > 0.0:
            ex += float(np.random.normal(0, self.loc_jitter))
            ey += float(np.random.normal(0, self.loc_jitter))
        m2o = TransformStamped()
        m2o.header.stamp = od.header.stamp
        m2o.header.frame_id = "map"
        m2o.child_frame_id = "odom"
        m2o.transform.translation.x = ex
        m2o.transform.translation.y = ey
        m2o.transform.rotation.z = math.sin(self.loc_bias[2] / 2.0)
        m2o.transform.rotation.w = math.cos(self.loc_bias[2] / 2.0)
        self.tf_bcast.sendTransform(m2o)

        # odom -> base_footprint (ground truth)
        tf = TransformStamped()
        tf.header.stamp = od.header.stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = self.rx
        tf.transform.translation.y = self.ry
        tf.transform.rotation.z = math.sin(self.rth / 2.0)
        tf.transform.rotation.w = math.cos(self.rth / 2.0)
        self.tf_bcast.sendTransform(tf)

        # render + publish image (raw + compressed)
        img = self._render()
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = od.header.stamp
        msg.header.frame_id = "camera_frame"
        self.pub_img.publish(msg)
        cmsg = self.bridge.cv2_to_compressed_imgmsg(img, dst_format="jpg")
        cmsg.header = msg.header
        self.pub_img_c.publish(cmsg)

        # Emulate the on-Jetson /qr/pose + /qr/data (camera->QR, +X out of face).
        if self.publish_qr_pose:
            self._publish_qr_pose(od.header.stamp)

        if (now - self.last_log) > 1.0 and not self.done:
            self.last_log = now
            self.get_logger().info(
                f"robot x={self.rx:.2f} y={self.ry:.2f} th={math.degrees(self.rth):.0f}deg "
                f"| cmd v={self.v:.3f} w={self.w:.3f}")


def main():
    rclpy.init()
    node = QRSimWorld()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
