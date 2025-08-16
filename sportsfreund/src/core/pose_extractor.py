"""
Core Pose Extraction using MediaPipe
Extracts normalized pose landmarks from video frames
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_pose_data(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract pose landmarks and calculate angles/positions from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract normalized coordinates
        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }

        # Calculate key angles and positions
        angles = self._calculate_key_angles(landmarks)
        positions = self._calculate_key_positions(landmarks)

        return {
            'landmarks': landmarks,
            'angles': angles,
            'positions': positions,
            'frame_data': results
        }

    def _calculate_key_angles(self, landmarks: Dict) -> Dict[str, float]:
        """Calculate important joint angles"""
        angles = {}

        try:
            # MediaPipe landmark indices
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14
            LEFT_WRIST = 15
            RIGHT_WRIST = 16

            # Knee angles (hip-knee-ankle)
            angles['left_knee'] = self._angle_between_points(
                landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]
            )
            angles['right_knee'] = self._angle_between_points(
                landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]
            )

            # Hip angles (shoulder-hip-knee)
            angles['left_hip'] = self._angle_between_points(
                landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_KNEE]
            )
            angles['right_hip'] = self._angle_between_points(
                landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE]
            )

            # Elbow angles (shoulder-elbow-wrist)
            angles['left_elbow'] = self._angle_between_points(
                landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]
            )
            angles['right_elbow'] = self._angle_between_points(
                landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]
            )

            # Back angle (torso inclination)
            shoulder_center = {
                'x': (landmarks[LEFT_SHOULDER]['x'] + landmarks[RIGHT_SHOULDER]['x']) / 2,
                'y': (landmarks[LEFT_SHOULDER]['y'] + landmarks[RIGHT_SHOULDER]['y']) / 2,
                'z': (landmarks[LEFT_SHOULDER]['z'] + landmarks[RIGHT_SHOULDER]['z']) / 2
            }
            hip_center = {
                'x': (landmarks[LEFT_HIP]['x'] + landmarks[RIGHT_HIP]['x']) / 2,
                'y': (landmarks[LEFT_HIP]['y'] + landmarks[RIGHT_HIP]['y']) / 2,
                'z': (landmarks[LEFT_HIP]['z'] + landmarks[RIGHT_HIP]['z']) / 2
            }
            vertical_point = {
                'x': hip_center['x'],
                'y': hip_center['y'] - 0.1,
                'z': hip_center['z']
            }
            angles['back'] = self._angle_between_points(shoulder_center, hip_center, vertical_point)

        except Exception as e:
            # Return default angles if calculation fails
            angles = {
                'left_knee': 180.0,
                'right_knee': 180.0,
                'left_hip': 180.0,
                'right_hip': 180.0,
                'left_elbow': 180.0,
                'right_elbow': 180.0,
                'back': 180.0
            }

        return angles

    def _calculate_key_positions(self, landmarks: Dict) -> Dict[str, Dict]:
        """Calculate key body positions"""
        positions = {}

        try:
            # Key positions for movement analysis
            positions['hip_center'] = {
                'x': (landmarks[23]['x'] + landmarks[24]['x']) / 2,
                'y': (landmarks[23]['y'] + landmarks[24]['y']) / 2
            }
            positions['knee_center'] = {
                'x': (landmarks[25]['x'] + landmarks[26]['x']) / 2,
                'y': (landmarks[25]['y'] + landmarks[26]['y']) / 2
            }
            positions['shoulder_center'] = {
                'x': (landmarks[11]['x'] + landmarks[12]['x']) / 2,
                'y': (landmarks[11]['y'] + landmarks[12]['y']) / 2
            }

        except Exception as e:
            positions = {
                'hip_center': {'x': 0.5, 'y': 0.5},
                'knee_center': {'x': 0.5, 'y': 0.7},
                'shoulder_center': {'x': 0.5, 'y': 0.3}
            }

        return positions

    def _angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points"""
        try:
            # Vectors
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            return np.degrees(angle)
        except:
            return 180.0

    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if pose_data and 'frame_data' in pose_data:
            results = pose_data['frame_data']
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
        return frame
