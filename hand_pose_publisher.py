#!/usr/bin/env python3
"""
hand_pose_publisher.py – Gesture-based teleoperation publisher for Franka Panda.

Captures webcam via OpenCV, tracks hand via MediaPipe, detects gestures,
maps hand position to the Franka workspace, and publishes:
  • /franka/target_pose     (geometry_msgs/PoseStamped)  – EE target
  • /franka/gripper_command  (std_msgs/Float64)           – gripper width (0–0.04 m)
  • /franka/teleop_state     (std_msgs/String)            – current state name

Gesture commands:
  🖐️ Open Palm  → MOVE   (robot follows hand)
  ✊ Fist       → FREEZE (hold current position)
  🤏 Pinch      → PICK   (close gripper, keep following)
  ✋ Release    → PLACE  (open gripper, return to MOVE)
  ☝️ Point Up   → WRIST  (index finger angle → wrist rotation)
"""

import math
import os
import urllib.request
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String
import cv2
import mediapipe as mp


# ═══════════════════════════════════════════════════════════════════════
# Teleop State Machine
# ═══════════════════════════════════════════════════════════════════════
class TeleopState(Enum):
    IDLE    = "IDLE"      # No hand detected
    MOVE    = "MOVE"      # Open palm – robot follows hand
    FREEZE  = "FREEZE"    # Fist – hold position
    PICKED  = "PICKED"    # Pinched – gripper closed, still following
    WRIST   = "WRIST"     # Pointing – control wrist rotation


# ═══════════════════════════════════════════════════════════════════════
# Smoothing filter
# ═══════════════════════════════════════════════════════════════════════
class LowPassFilter:
    """Exponential Moving Average (EMA) filter for a single signal."""

    def __init__(self, alpha: float = 0.4):
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


# ═══════════════════════════════════════════════════════════════════════
# Gesture Detector
# ═══════════════════════════════════════════════════════════════════════
class GestureDetector:
    """Detects hand gestures from MediaPipe 21-landmark hand model.

    Landmark indices:
        0  = wrist
        4  = thumb tip,      3 = thumb IP
        8  = index tip,      6 = index PIP
        12 = middle tip,    10 = middle PIP
        16 = ring tip,      14 = ring PIP
        20 = pinky tip,     18 = pinky PIP
    """

    PINCH_CLOSE_THRESH = 0.05   # normalised distance to close gripper
    PINCH_OPEN_THRESH  = 0.08   # hysteresis band to open gripper

    def __init__(self):
        self.was_pinched = False

    def _is_finger_extended(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        """A finger is extended if its tip is above (lower Y) its PIP joint."""
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def _is_thumb_extended(self, landmarks) -> bool:
        """Thumb: tip (4) is further from wrist (0) than IP (3) in X."""
        return abs(landmarks[4].x - landmarks[0].x) > abs(landmarks[3].x - landmarks[0].x)

    def _finger_states(self, landmarks) -> dict:
        """Returns dict of finger_name → bool (extended or not)."""
        return {
            'thumb':  self._is_thumb_extended(landmarks),
            'index':  self._is_finger_extended(landmarks, 8, 6),
            'middle': self._is_finger_extended(landmarks, 12, 10),
            'ring':   self._is_finger_extended(landmarks, 16, 14),
            'pinky':  self._is_finger_extended(landmarks, 20, 18),
        }

    def _pinch_distance(self, landmarks) -> float:
        """Euclidean distance between thumb tip (4) and index tip (8)."""
        t, i = landmarks[4], landmarks[8]
        return math.sqrt((t.x - i.x)**2 + (t.y - i.y)**2 + (t.z - i.z)**2)

    def detect(self, landmarks) -> tuple:
        """Detect gesture and return (gesture_name, pinch_dist, finger_states).

        Returns:
            gesture:      str — 'OPEN_PALM', 'FIST', 'PINCH', 'POINT', 'OTHER'
            pinch_dist:   float
            finger_states: dict
        """
        fs = self._finger_states(landmarks)
        pinch_dist = self._pinch_distance(landmarks)

        extended_count = sum(fs.values())

        # ── Pinch detection with hysteresis ────────────────────────
        if not self.was_pinched and pinch_dist < self.PINCH_CLOSE_THRESH:
            self.was_pinched = True
        elif self.was_pinched and pinch_dist > self.PINCH_OPEN_THRESH:
            self.was_pinched = False

        if self.was_pinched:
            return 'PINCH', pinch_dist, fs

        # ── Fist: no fingers extended ──────────────────────────────
        if extended_count <= 1 and not fs['index']:
            return 'FIST', pinch_dist, fs

        # ── Point: only index extended ─────────────────────────────
        if fs['index'] and not fs['middle'] and not fs['ring'] and not fs['pinky']:
            return 'POINT', pinch_dist, fs

        # ── Open palm: 4+ fingers extended ─────────────────────────
        if extended_count >= 4:
            return 'OPEN_PALM', pinch_dist, fs

        return 'OTHER', pinch_dist, fs


# ═══════════════════════════════════════════════════════════════════════
# Main ROS 2 Node
# ═══════════════════════════════════════════════════════════════════════
class HandPosePublisher(Node):
    def __init__(self):
        super().__init__('hand_pose_publisher')

        # ── Publishers ────────────────────────────────────────────────
        self.pose_pub    = self.create_publisher(PoseStamped, '/franka/target_pose', 10)
        self.gripper_pub = self.create_publisher(Float64, '/franka/gripper_command', 10)
        self.state_pub   = self.create_publisher(String, '/franka/teleop_state', 10)

        # ── State machine ─────────────────────────────────────────────
        self.state = TeleopState.IDLE
        self.gesture_detector = GestureDetector()

        # ── Gripper ───────────────────────────────────────────────────
        self.gripper_width = 0.04       # metres (0.04 = fully open, 0.0 = closed)
        self.GRIPPER_OPEN  = 0.04
        self.GRIPPER_CLOSED = 0.0

        # ── EMA smoothing (faster alpha = more responsive) ────────────
        self.filter_x = LowPassFilter(alpha=0.4)
        self.filter_y = LowPassFilter(alpha=0.4)
        self.filter_z = LowPassFilter(alpha=0.4)

        # ── Last known MAPPED robot position ──────────────────────────
        self.last_robot_x = 0.45
        self.last_robot_y = 0.0
        self.last_robot_z = 0.4

        # ── Workspace mapping (AMPLIFIED) ─────────────────────────────
        # Robot workspace in metres
        self.ROBOT_X_RANGE = [0.15, 0.75]   # forward / backward
        self.ROBOT_Y_RANGE = [-0.5, 0.5]    # left / right
        self.ROBOT_Z_RANGE = [0.05, 0.75]   # up / down

        # Camera input ranges — trimmed edges for bigger movement
        self.MP_XY_RANGE = [0.15, 0.85]    # ignore outer 15% of frame
        self.MP_Z_RANGE  = [-0.15, 0.15]   # more sensitive depth

        # ── End-effector orientation ──────────────────────────────────
        # Default: gripper pointing downward (90° about Y)
        self.ee_orient = [0.0, 0.7071068, 0.0, 0.7071068]  # x, y, z, w

        # ── Wrist rotation (for POINT gesture) ────────────────────────
        self.wrist_angle = 0.0   # radians, mapped to EE Z-rotation

        # ── MediaPipe initialisation ──────────────────────────────────
        self.use_tasks_api = False
        self.hands = None
        self.hand_landmarker = None
        self.hand_connections = (
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        )

        if hasattr(mp, 'solutions'):
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands   = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
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
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.7,
                    min_tracking_confidence=0.7,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            else:
                self.get_logger().error('Hand Landmarker model unavailable.')

        # ── OpenCV capture ────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open webcam on device 0")

        # ── Timer at 30 Hz ────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info(
            'HandPosePublisher started — gesture teleoperation active at 30 Hz'
        )

    # ──────────────────────────────────────────────────────────────────
    # Coordinate mapping (amplified)
    # ──────────────────────────────────────────────────────────────────
    def _map_to_robot_frame(self, mp_x: float, mp_y: float, mp_z: float):
        """MediaPipe normalised wrist coords → smoothed Panda workspace (metres)."""
        sx = self.filter_x.update(mp_x)
        sy = self.filter_y.update(mp_y)
        sz = self.filter_z.update(mp_z)

        # Camera X → Robot Y  (invert: screen-left = robot +Y)
        robot_y = float(np.interp(sx, self.MP_XY_RANGE,
                                  [self.ROBOT_Y_RANGE[1], self.ROBOT_Y_RANGE[0]]))

        # Camera Y → Robot Z  (invert: screen-top = higher Z)
        robot_z = float(np.interp(sy, self.MP_XY_RANGE,
                                  [self.ROBOT_Z_RANGE[1], self.ROBOT_Z_RANGE[0]]))

        # Camera Z (depth) → Robot X
        robot_x = float(np.interp(sz, self.MP_Z_RANGE, self.ROBOT_X_RANGE))

        return robot_x, robot_y, robot_z

    # ──────────────────────────────────────────────────────────────────
    # State machine transitions
    # ──────────────────────────────────────────────────────────────────
    def _update_state(self, gesture: str):
        """Transition the teleop state based on detected gesture."""
        prev = self.state

        if gesture == 'FIST':
            self.state = TeleopState.FREEZE

        elif gesture == 'PINCH':
            # Close gripper — enter PICKED (can still move)
            self.gripper_width = self.GRIPPER_CLOSED
            self.state = TeleopState.PICKED

        elif gesture == 'OPEN_PALM':
            if prev == TeleopState.PICKED:
                # Was holding → release = PLACE → immediately back to MOVE
                self.gripper_width = self.GRIPPER_OPEN
            self.state = TeleopState.MOVE

        elif gesture == 'POINT':
            self.state = TeleopState.WRIST

        elif gesture == 'OTHER':
            # Keep previous state if gesture is ambiguous
            pass

        # Log transitions
        if self.state != prev:
            self.get_logger().info(f'State: {prev.value} → {self.state.value}')

    # ──────────────────────────────────────────────────────────────────
    # Wrist rotation from pointing gesture
    # ──────────────────────────────────────────────────────────────────
    def _compute_wrist_rotation(self, landmarks):
        """Map the angle of the index finger to wrist rotation (EE Z-axis)."""
        # Vector from wrist (0) to index MCP (5)
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        dx = index_mcp.x - wrist.x
        dy = index_mcp.y - wrist.y
        angle = math.atan2(dy, dx)  # radians, ~-π to π

        # Map to a wrist rotation quaternion (rotate about Z)
        # Smooth it
        self.wrist_angle = 0.4 * angle + 0.6 * self.wrist_angle

        # Build quaternion: base downward (Y-90°) + Z-rotation
        cy = math.cos(self.wrist_angle / 2)
        sy_q = math.sin(self.wrist_angle / 2)

        # Combined: R_y(90°) * R_z(wrist_angle)
        # Base quat: (0, 0.707, 0, 0.707)
        # Rotation about local Z: (0, 0, sin(a/2), cos(a/2))
        # Combined via quaternion multiplication:
        bx, by, bz, bw = 0.0, 0.7071068, 0.0, 0.7071068
        rx, ry, rz, rw = 0.0, 0.0, sy_q, cy

        # q_result = q_base * q_rot
        ox = bw*rx + bx*rw + by*rz - bz*ry
        oy = bw*ry - bx*rz + by*rw + bz*rx
        oz = bw*rz + bx*ry - by*rx + bz*rw
        ow = bw*rw - bx*rx - by*ry - bz*rz

        self.ee_orient = [ox, oy, oz, ow]

    # ──────────────────────────────────────────────────────────────────
    # Timer callback (30 Hz)
    # ──────────────────────────────────────────────────────────────────
    def timer_callback(self):
        if not self.cap.isOpened():
            return

        success, image = self.cap.read()
        if not success:
            self.get_logger().warn("Ignoring empty camera frame.")
            return

        image = cv2.flip(image, 1)  # mirror
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # ── Build PoseStamped ─────────────────────────────────────────
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'panda_link0'

        hand_detected = False
        gesture_name = 'NONE'
        pinch_dist = 0.0
        finger_states = {}

        # ── MediaPipe processing ──────────────────────────────────────
        landmarks_list = None

        if self.use_tasks_api:
            if self.hand_landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = self.hand_landmarker.detect(mp_image)
                if result.hand_landmarks:
                    landmarks_list = result.hand_landmarks[0]
                    hand_detected = True
                    self._draw_hand_landmarks(image_bgr, landmarks_list)
        else:
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Convert to same format as tasks API
                landmarks_list = hand_landmarks.landmark
                hand_detected = True

                self.mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # ── Process landmarks ─────────────────────────────────────────
        if hand_detected and landmarks_list is not None:
            # Detect gesture
            gesture_name, pinch_dist, finger_states = self.gesture_detector.detect(landmarks_list)

            # Update state machine
            self._update_state(gesture_name)

            # Map position (always compute, even if frozen — we just don't publish in FREEZE)
            wrist = landmarks_list[0]
            rx, ry, rz = self._map_to_robot_frame(wrist.x, wrist.y, wrist.z)

            if self.state in (TeleopState.MOVE, TeleopState.PICKED, TeleopState.WRIST):
                self.last_robot_x, self.last_robot_y, self.last_robot_z = rx, ry, rz

            # Handle WRIST rotation
            if self.state == TeleopState.WRIST:
                self._compute_wrist_rotation(landmarks_list)
            else:
                # Reset to default downward orientation
                self.ee_orient = [0.0, 0.7071068, 0.0, 0.7071068]

        else:
            self.state = TeleopState.IDLE

        # ── Set pose message ──────────────────────────────────────────
        msg.pose.position.x = self.last_robot_x
        msg.pose.position.y = self.last_robot_y
        msg.pose.position.z = self.last_robot_z

        msg.pose.orientation.x = self.ee_orient[0]
        msg.pose.orientation.y = self.ee_orient[1]
        msg.pose.orientation.z = self.ee_orient[2]
        msg.pose.orientation.w = self.ee_orient[3]

        self.pose_pub.publish(msg)

        # ── Gripper command ───────────────────────────────────────────
        gripper_msg = Float64()
        gripper_msg.data = self.gripper_width
        self.gripper_pub.publish(gripper_msg)

        # ── State topic ───────────────────────────────────────────────
        state_msg = String()
        state_msg.data = self.state.value
        self.state_pub.publish(state_msg)

        # ── HUD overlay ──────────────────────────────────────────────
        self._draw_hud(image_bgr, hand_detected, gesture_name, pinch_dist, finger_states)

        cv2.imshow('Hand Pose Publisher', image_bgr)
        cv2.waitKey(1)

    # ──────────────────────────────────────────────────────────────────
    # HUD Drawing
    # ──────────────────────────────────────────────────────────────────
    def _draw_hud(self, image, hand_detected, gesture, pinch_dist, finger_states):
        """Draw a comprehensive heads-up display on the camera feed."""
        h, w = image.shape[:2]

        # ── Background panel ──────────────────────────────────────
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (320, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        y = 25

        # ── State ─────────────────────────────────────────────────
        state_colors = {
            TeleopState.IDLE:   (128, 128, 128),
            TeleopState.MOVE:   (0, 255, 0),
            TeleopState.FREEZE: (0, 165, 255),
            TeleopState.PICKED: (0, 0, 255),
            TeleopState.WRIST:  (255, 255, 0),
        }
        color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(image, f"STATE: {self.state.value}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28

        # ── Gesture ───────────────────────────────────────────────
        gesture_icons = {
            'OPEN_PALM': '[ PALM ]',
            'FIST':      '[ FIST ]',
            'PINCH':     '[ PINCH]',
            'POINT':     '[ POINT]',
            'OTHER':     '[  ...  ]',
            'NONE':      '[  ---  ]',
        }
        cv2.putText(image, f"Gesture: {gesture_icons.get(gesture, gesture)}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25

        # ── Position ──────────────────────────────────────────────
        cv2.putText(image,
                    f"X:{self.last_robot_x:.3f}  Y:{self.last_robot_y:.3f}  Z:{self.last_robot_z:.3f}",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 25

        # ── Gripper ───────────────────────────────────────────────
        grip_text = "CLOSED" if self.gripper_width < 0.01 else f"OPEN ({self.gripper_width:.3f}m)"
        grip_color = (0, 0, 255) if self.gripper_width < 0.01 else (0, 255, 0)
        cv2.putText(image, f"Gripper: {grip_text}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 1)
        y += 25

        # ── Pinch distance ────────────────────────────────────────
        if hand_detected:
            cv2.putText(image, f"Pinch dist: {pinch_dist:.3f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            y += 22

            # ── Finger states ─────────────────────────────────────
            if finger_states:
                fingers = ''.join([
                    '👍' if finger_states.get('thumb') else '·',
                    '☝' if finger_states.get('index') else '·',
                    '|' if finger_states.get('middle') else '·',
                    '|' if finger_states.get('ring') else '·',
                    '|' if finger_states.get('pinky') else '·',
                ])
                cv2.putText(image, f"Fingers: {fingers}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            cv2.putText(image, "NO HAND DETECTED", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # ── Gesture guide (bottom of screen) ──────────────────────
        guide = "Palm=MOVE  Fist=FREEZE  Pinch=PICK  Release=PLACE  Point=WRIST"
        cv2.putText(image, guide, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────
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
            self.get_logger().info('Downloading MediaPipe hand landmarker model…')
            urllib.request.urlretrieve(model_url, model_path)
            return model_path
        except Exception as exc:
            self.get_logger().error(f'Model download failed: {exc}')
            return None

    def _draw_hand_landmarks(self, image, landmarks):
        h, w = image.shape[:2]
        for s, e in self.hand_connections:
            sp = (int(landmarks[s].x * w), int(landmarks[s].y * h))
            ep = (int(landmarks[e].x * w), int(landmarks[e].y * h))
            cv2.line(image, sp, ep, (0, 255, 0), 2)
        for lm in landmarks:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = HandPosePublisher()

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
