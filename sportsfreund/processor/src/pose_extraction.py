"""
Pose-Extraktion aus Videos und Live-Webcam-Streams mittels MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Tuple, List, Dict, Any, Optional

from .config import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE
)

class PoseExtractor:
    """Klasse zur Extraktion von Pose-Daten aus Videos und Live-Streams."""

    def __init__(self):
        """Initialisierung des MediaPipe Pose-Detektors."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )

        # Index der Schlüsselpunkte für einfacheren Zugriff
        self.keypoint_indices = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }

    def extract_pose_from_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrahiert Pose-Daten aus einem einzelnen Frame.

        Args:
            frame: Ein BGR-Bild als NumPy-Array

        Returns:
            Tuple mit:
                - Das Bild mit eingezeichneten Posen-Landmarks
                - NumPy-Array der Pose-Daten [num_landmarks, 3] (x, y, visibility)
        """
        # Konvertiere BGR zu RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deaktiviere Schreiben in das Bild
        frame_rgb.flags.writeable = False

        # Verarbeite das Bild
        results = self.pose.process(frame_rgb)

        # Aktiviere Schreiben in das Bild wieder
        frame_rgb.flags.writeable = True

        # Konvertiere zurück zu BGR
        frame_with_landmarks = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Zeichne die Pose-Landmarks
            self.mp_drawing.draw_landmarks(
                frame_with_landmarks,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Extrahiere Pose-Daten
            pose_data = np.array([[landmark.x, landmark.y, landmark.visibility]
                                for landmark in results.pose_landmarks.landmark],
                                dtype=np.float32)
            return frame_with_landmarks, pose_data

        # Wenn keine Pose erkannt wurde
        return frame_with_landmarks, np.zeros((33, 3), dtype=np.float32)

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[np.ndarray]:
        """
        Verarbeitet ein Video und extrahiert Pose-Daten.

        Args:
            video_path: Pfad zum Eingabevideo
            output_path: Pfad zum Speichern des verarbeiteten Videos (optional)

        Returns:
            Liste der Pose-Daten für jeden Frame
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Konnte das Video nicht öffnen: {video_path}")

        # Video-Eigenschaften
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialisiere Videoschreiber, wenn ein Ausgabepfad angegeben ist
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_poses = []

        try:
            frame_idx = 0
            start_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extrahiere Pose-Daten
                frame_with_landmarks, pose_data = self.extract_pose_from_frame(frame)
                all_poses.append(pose_data)

                # Schreibe Frame, wenn ein Ausgabepfad angegeben ist
                if writer:
                    writer.write(frame_with_landmarks)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    elapsed_time = time.time() - start_time
                    frames_per_second = frame_idx / elapsed_time
                    print(f"Verarbeitet: {frame_idx}/{total_frames} Frames ({frames_per_second:.2f} FPS)")

        finally:
            cap.release()
            if writer:
                writer.release()

        return all_poses

    def extract_features(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Extrahiert zusätzliche Merkmale aus den Pose-Daten.

        Diese Funktion kann erweitert werden, um abgeleitete Merkmale wie Gelenkwinkel,
        relative Positionen usw. zu berechnen.

        Args:
            pose_data: NumPy-Array der Pose-Daten [num_landmarks, 3]

        Returns:
            NumPy-Array der extrahierten Merkmale
        """
        # Beispiel: Berechne Gelenkwinkel für Knie und Ellbogen
        features = []

        # Berechne Winkel für linkes Knie (Hip-Knee-Ankle)
        if pose_data.size > 0:  # Prüfe, ob Pose-Daten vorhanden sind
            # Linkes Knie
            left_hip = pose_data[self.keypoint_indices['left_hip']][:2]
            left_knee = pose_data[self.keypoint_indices['left_knee']][:2]
            left_ankle = pose_data[self.keypoint_indices['left_ankle']][:2]
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)

            # Rechtes Knie
            right_hip = pose_data[self.keypoint_indices['right_hip']][:2]
            right_knee = pose_data[self.keypoint_indices['right_knee']][:2]
            right_ankle = pose_data[self.keypoint_indices['right_ankle']][:2]
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)

            # Linker Ellbogen
            left_shoulder = pose_data[self.keypoint_indices['left_shoulder']][:2]
            left_elbow = pose_data[self.keypoint_indices['left_elbow']][:2]
            left_wrist = pose_data[self.keypoint_indices['left_wrist']][:2]
            left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Rechter Ellbogen
            right_shoulder = pose_data[self.keypoint_indices['right_shoulder']][:2]
            right_elbow = pose_data[self.keypoint_indices['right_elbow']][:2]
            right_wrist = pose_data[self.keypoint_indices['right_wrist']][:2]
            right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Rückenneigung (Winkel zwischen Hüfte und Schulter zur Vertikalen)
            mid_hip = (pose_data[self.keypoint_indices['left_hip']][:2] +
                       pose_data[self.keypoint_indices['right_hip']][:2]) / 2
            mid_shoulder = (pose_data[self.keypoint_indices['left_shoulder']][:2] +
                            pose_data[self.keypoint_indices['right_shoulder']][:2]) / 2
            vertical = np.array([mid_hip[0], 0])
            back_angle = self._calculate_angle(vertical, mid_hip, mid_shoulder)

            features = [left_knee_angle, right_knee_angle, left_elbow_angle, right_elbow_angle, back_angle]

        return np.array(features, dtype=np.float32)

    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Berechnet den Winkel zwischen drei Punkten in Grad.

        Args:
            a: Erster Punkt
            b: Mittlerer Punkt (Scheitelpunkt)
            c: Dritter Punkt

        Returns:
            Winkel in Grad
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle)

        return angle_deg

    def start_webcam(self, camera_id: int = 0):
        """
        Startet die Webcam und verarbeitet Frames in Echtzeit.

        Args:
            camera_id: ID der zu verwendenden Kamera

        Returns:
            Generator, der Tupel mit (Frame mit Landmarks, Pose-Daten) zurückgibt
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Konnte die Webcam mit ID {camera_id} nicht öffnen")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extrahiere Pose-Daten
                frame_with_landmarks, pose_data = self.extract_pose_from_frame(frame)

                yield frame_with_landmarks, pose_data

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
