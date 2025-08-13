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
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract pose landmarks from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract normalized coordinates
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # Calculate key angles and positions
        coords = np.array(landmarks).reshape(-1, 4)
        
        return {
            'raw_landmarks': np.array(landmarks),
            'coordinates': coords,
            'angles': self._calculate_key_angles(coords),
            'positions': self._calculate_key_positions(coords),
            'frame_data': results
        }
    
    def _calculate_key_angles(self, coords: np.ndarray) -> Dict[str, float]:
        """Calculate important joint angles"""
        angles = {}
        
        try:
            # Knee angles (hip-knee-ankle)
            angles['left_knee'] = self._angle_between_points(
                coords[23][:3], coords[25][:3], coords[27][:3]  # left_hip, left_knee, left_ankle
            )
            angles['right_knee'] = self._angle_between_points(
                coords[24][:3], coords[26][:3], coords[28][:3]  # right_hip, right_knee, right_ankle
            )
            
            # Hip angles (shoulder-hip-knee)
            angles['left_hip'] = self._angle_between_points(
                coords[11][:3], coords[23][:3], coords[25][:3]  # left_shoulder, left_hip, left_knee
            )
            angles['right_hip'] = self._angle_between_points(
                coords[12][:3], coords[24][:3], coords[26][:3]  # right_shoulder, right_hip, right_knee
            )
            
            # Back angle (torso inclination)
            shoulder_center = (coords[11][:3] + coords[12][:3]) / 2
            hip_center = (coords[23][:3] + coords[24][:3]) / 2
            vertical = hip_center + [0, 0.1, 0]
            angles['back'] = self._angle_between_points(shoulder_center, hip_center, vertical)
            
        except Exception:
            # Return default angles if calculation fails
            angles = {k: 0.0 for k in ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'back']}
        
        return angles
    
    def _calculate_key_positions(self, coords: np.ndarray) -> Dict[str, float]:
        """Calculate important body positions"""
        positions = {}
        
        try:
            # Heights (Y coordinates)
            positions['hip_height'] = (coords[23][1] + coords[24][1]) / 2
            positions['knee_height'] = (coords[25][1] + coords[26][1]) / 2
            positions['ankle_height'] = (coords[27][1] + coords[28][1]) / 2
            
            # Distances
            positions['knee_distance'] = abs(coords[25][0] - coords[26][0])  # knee separation
            positions['ankle_distance'] = abs(coords[27][0] - coords[28][0])  # foot separation
            
        except Exception:
            positions = {k: 0.0 for k in ['hip_height', 'knee_height', 'ankle_height', 'knee_distance', 'ankle_distance']}
        
        return positions
    
    def _angle_between_points(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle at point b between points a and c"""
        try:
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 0.0
    
    def visualize_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if pose_data and 'frame_data' in pose_data:
            self.mp_drawing.draw_landmarks(
                frame, 
                pose_data['frame_data'].pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
