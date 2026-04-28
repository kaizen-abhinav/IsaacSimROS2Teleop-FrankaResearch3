#!/usr/bin/env python3

import math
import os
import urllib.request

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import cv2
import mediapipe as mp


class LowPassFilter:
    """Exponential Moving Average (EMA) filter for smoothing a single signal."""

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor in (0, 1]. Lower = smoother but more lag.
                   Higher = less smoothing but more responsive.
        """
        self.alpha = alpha
        self._value = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None


def scale_linear(value: float, in_min: float, in_max: float,
                 out_min: float, out_max: float) -> float:
    """Linearly map *value* from [in_min, in_max] to [out_min, out_max],
    clamping to the output range."""
    normalized = (value - in_min) / (in_max - in_min)
    normalized = max(0.0, min(1.0, normalized))      # clamp to [0, 1]
    return out_min + normalized * (out_max - out_min)


class VisionTeleopNode(Node):
    def __init__(self):
        super().__init__('vision_teleop_node')

        # Initialize publishers
        self.publisher_ = self.create_publisher(PoseStamped, '/franka/target_pose', 10)
        self.gripper_pub = self.create_publisher(Bool, '/franka/gripper_command', 10)

        # ── Pinch detection ───────────────────────────────────────────
        # Euclidean distance threshold (normalised coords) below which
        # we consider thumb-tip and index-tip to be "pinching".
        self.PINCH_THRESHOLD = 0.05
        self.gripper_closed = False

        # ── Smoothing filters (one per axis) ──────────────────────────
        # alpha=0.25 gives a good balance between responsiveness and jitter
        self.filter_x = LowPassFilter(alpha=0.25)
        self.filter_y = LowPassFilter(alpha=0.25)
        self.filter_z = LowPassFilter(alpha=0.25)

        # ── Workspace mapping constants ───────────────────────────────
        # MediaPipe normalised range for X and Y is [0.0, 1.0].
        # Z is a relative depth value that typically falls in [-0.3, 0.3].
        self.MP_X_MIN, self.MP_X_MAX = 0.0, 1.0
        self.MP_Y_MIN, self.MP_Y_MAX = 0.0, 1.0
        self.MP_Z_MIN, self.MP_Z_MAX = -0.3, 0.3

        # Physical robot workspace in metres (Panda reach ≈ 0.855 m)
        # Camera X  → Robot Y  (left / right)
        self.ROBOT_Y_MIN, self.ROBOT_Y_MAX = -0.4, 0.4
        # Camera Y  → Robot Z  (up / down)
        self.ROBOT_Z_MIN, self.ROBOT_Z_MAX =  0.1, 0.6
        # Camera Z  → Robot X  (forward / backward)
        self.ROBOT_X_MIN, self.ROBOT_X_MAX =  0.3, 0.7

        # ── MediaPipe initialisation ──────────────────────────────────
        self.use_tasks_api = False
        self.hands = None
        self.hand_landmarker = None
        self.hand_connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        )

        if hasattr(mp, 'solutions'):
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
        else:
            self.use_tasks_api = True
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_hand_landmarker_model()
            if model_path:
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    num_hands=1,
                    min_hand_detection_confidence=0.8,
                    min_hand_presence_confidence=0.8,
                    min_tracking_confidence=0.8,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            else:
                self.get_logger().error(
                    'Hand Landmarker model unavailable; MediaPipe tracking disabled.'
                )

        # ── OpenCV capture ────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open OpenCV VideoCapture on device 0")

        # Timer at 30 Hz
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Vision Teleop Node initialised – smoothing + workspace mapping active.')

    # ──────────────────────────────────────────────────────────────────
    # Coordinate helpers
    # ──────────────────────────────────────────────────────────────────
    def _map_to_robot_frame(self, mp_x: float, mp_y: float, mp_z: float):
        """Convert raw MediaPipe normalised coords → smoothed robot-frame metres."""
        # 1. Smooth the raw values
        sx = self.filter_x.update(mp_x)
        sy = self.filter_y.update(mp_y)
        sz = self.filter_z.update(mp_z)

        # 2. Linear scale + axis swap
        robot_x = scale_linear(sz, self.MP_Z_MIN, self.MP_Z_MAX,
                               self.ROBOT_X_MIN, self.ROBOT_X_MAX)
        robot_y = scale_linear(sx, self.MP_X_MIN, self.MP_X_MAX,
                               self.ROBOT_Y_MIN, self.ROBOT_Y_MAX)
        robot_z = scale_linear(sy, self.MP_Y_MIN, self.MP_Y_MAX,
                               self.ROBOT_Z_MIN, self.ROBOT_Z_MAX)

        # Invert Y so that moving hand right → positive robot Y
        robot_y = -robot_y

        # Invert Z so that raising hand → higher Z
        robot_z = self.ROBOT_Z_MAX - (robot_z - self.ROBOT_Z_MIN)

        return robot_x, robot_y, robot_z

    # ──────────────────────────────────────────────────────────────────
    # Main timer callback
    # ──────────────────────────────────────────────────────────────────
    def timer_callback(self):
        if not self.cap.isOpened():
            return

        success, image = self.cap.read()
        if not success:
            self.get_logger().warn("Ignoring empty camera frame.")
            return

        # Mirror for selfie-view
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        hand_detected = False
        pinch_distance = None

        if self.use_tasks_api:
            if self.hand_landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = self.hand_landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    wrist = lm[0]
                    rx, ry, rz = self._map_to_robot_frame(wrist.x, wrist.y, wrist.z)
                    msg.pose.position.x = rx
                    msg.pose.position.y = ry
                    msg.pose.position.z = rz
                    hand_detected = True

                    # Pinch: THUMB_TIP (4) vs INDEX_FINGER_TIP (8)
                    pinch_distance = self._euclidean_3d(lm[4], lm[8])

                    self._draw_hand_landmarks(image_bgr, lm)
        else:
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                rx, ry, rz = self._map_to_robot_frame(wrist.x, wrist.y, wrist.z)
                msg.pose.position.x = rx
                msg.pose.position.y = ry
                msg.pose.position.z = rz
                hand_detected = True

                # Pinch: THUMB_TIP (4) vs INDEX_FINGER_TIP (8)
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                pinch_distance = self._euclidean_3d(thumb, index)

                self.mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        # ── Gripper logic ─────────────────────────────────────────────
        if pinch_distance is not None:
            self.gripper_closed = pinch_distance < self.PINCH_THRESHOLD

        gripper_msg = Bool()
        gripper_msg.data = self.gripper_closed
        self.gripper_pub.publish(gripper_msg)

        if not hand_detected:
            # Hold last filtered position instead of jumping to origin
            msg.pose.position.x = self.filter_x._value if self.filter_x._value is not None else 0.0
            msg.pose.position.y = self.filter_y._value if self.filter_y._value is not None else 0.0
            msg.pose.position.z = self.filter_z._value if self.filter_z._value is not None else 0.0

        self.publisher_.publish(msg)

        # ── HUD overlay ──────────────────────────────────────────────
        hud = (f"Robot X:{msg.pose.position.x:.3f}  "
               f"Y:{msg.pose.position.y:.3f}  "
               f"Z:{msg.pose.position.z:.3f}")
        cv2.putText(image_bgr, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        grip_label = "GRIP: CLOSED" if self.gripper_closed else "GRIP: OPEN"
        grip_colour = (0, 0, 255) if self.gripper_closed else (0, 255, 0)
        cv2.putText(image_bgr, grip_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, grip_colour, 2)

        if pinch_distance is not None:
            cv2.putText(image_bgr, f"Pinch: {pinch_distance:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if not hand_detected:
            cv2.putText(image_bgr, "NO HAND", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('MediaPipe Hand Tracking', image_bgr)
        cv2.waitKey(1)

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _euclidean_3d(a, b) -> float:
        """3D Euclidean distance between two landmarks."""
        return math.sqrt((a.x - b.x) ** 2 +
                         (a.y - b.y) ** 2 +
                         (a.z - b.z) ** 2)

    def _ensure_hand_landmarker_model(self):
        model_env_path = os.getenv('MP_HAND_LANDMARKER_MODEL')
        if model_env_path and os.path.isfile(model_env_path):
            return model_env_path

        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'mediapipe')
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, 'hand_landmarker.task')
        if os.path.isfile(model_path):
            return model_path

        model_url = (
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/latest/hand_landmarker.task'
        )
        try:
            self.get_logger().info(
                'Downloading MediaPipe hand landmarker model to cache.'
            )
            urllib.request.urlretrieve(model_url, model_path)
            return model_path
        except Exception as exc:
            self.get_logger().error(
                f'Failed to download hand landmarker model: {exc}'
            )
            return None

    def _draw_hand_landmarks(self, image, landmarks):
        height, width = image.shape[:2]

        for start_idx, end_idx in self.hand_connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_pt = (int(start.x * width), int(start.y * height))
            end_pt = (int(end.x * width), int(end.y * height))
            cv2.line(image, start_pt, end_pt, (0, 255, 0), 2)

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)


def main(args=None):
    rclpy.init(args=args)
    node = VisionTeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.hand_landmarker is not None:
            node.hand_landmarker.close()
        if node.hands is not None:
            node.hands.close()
        node.cap.release()
        cv2.destroyAllWindows()

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
