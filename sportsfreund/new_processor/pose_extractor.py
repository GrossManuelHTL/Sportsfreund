import cv2
import mediapipe as mp
import numpy as np
import time

class PoseExtractor:
    """
    Klasse zur Extraktion von Körperposen mit MediaPipe.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialisiert den PoseExtractor mit MediaPipe.

        Args:
            min_detection_confidence: Minimale Erkennungskonfidenz für MediaPipe
            min_tracking_confidence: Minimale Tracking-Konfidenz für MediaPipe
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Mapping von Gelenknamen zu MediaPipe-Indizes
        self.landmark_mapping = {
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

    def extract_landmarks(self, frame):
        """
        Extrahiert Landmarken aus einem Bild.

        Args:
            frame: Eingangsframe (BGR)

        Returns:
            landmarks: Landmarken-Objekt
            processed_frame: Frame mit eingezeichneten Landmarken
        """
        # Bildverarbeitung für MediaPipe (RGB konvertieren)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose-Schätzung mit MediaPipe
        results = self.pose.process(frame_rgb)

        # Frame mit Landmarken annotieren
        processed_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Koordinaten für die Winkelberechnung abrufen
            img_height, img_width, _ = frame.shape
            coords = self.get_landmark_coordinates(results.pose_landmarks, img_width, img_height)

            # Winkel berechnen
            joint_angles = self.calculate_joint_angles(coords)

            # Winkel auf dem Bild anzeigen
            if joint_angles and coords:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 255, 0)  # Grün
                thickness = 2

                # Kniewinkel anzeigen
                if 'left_knee_angle' in joint_angles and 'left_knee' in coords:
                    pos = (coords['left_knee'][0] + 10, coords['left_knee'][1])
                    text = f"{joint_angles['left_knee_angle']:.1f}°"
                    cv2.putText(processed_frame, text, pos, font, font_scale, color, thickness)

                if 'right_knee_angle' in joint_angles and 'right_knee' in coords:
                    pos = (coords['right_knee'][0] + 10, coords['right_knee'][1])
                    text = f"{joint_angles['right_knee_angle']:.1f}°"
                    cv2.putText(processed_frame, text, pos, font, font_scale, color, thickness)

                # Hüftwinkel anzeigen
                if 'left_hip_angle' in joint_angles and 'left_hip' in coords:
                    pos = (coords['left_hip'][0] + 10, coords['left_hip'][1])
                    text = f"{joint_angles['left_hip_angle']:.1f}°"
                    cv2.putText(processed_frame, text, pos, font, font_scale, color, thickness)

                if 'right_hip_angle' in joint_angles and 'right_hip' in coords:
                    pos = (coords['right_hip'][0] + 10, coords['right_hip'][1])
                    text = f"{joint_angles['right_hip_angle']:.1f}°"
                    cv2.putText(processed_frame, text, pos, font, font_scale, color, thickness)

                # Rückenwinkel anzeigen, oben auf dem Bild
                if 'back_angle' in joint_angles:
                    pos = (20, 30)  # Position oben links
                    text = f"Rücken: {joint_angles['back_angle']:.1f}°"
                    cv2.putText(processed_frame, text, pos, font, font_scale, color, thickness)

        return results.pose_landmarks, processed_frame

    def get_landmark_coordinates(self, landmarks, img_width, img_height):
        """
        Extrahiert die Koordinaten aller Landmarken.

        Args:
            landmarks: Mediapipe-Landmarken
            img_width: Bildbreite
            img_height: Bildhöhe

        Returns:
            dict: Dictionary mit Landmarken-Koordinaten
        """
        if not landmarks:
            return None

        coords = {}
        for name, idx in self.landmark_mapping.items():
            landmark = landmarks.landmark[idx]
            # Normalisierte Koordinaten in Pixel umwandeln
            coords[name] = (
                int(landmark.x * img_width),
                int(landmark.y * img_height),
                round(landmark.visibility, 2)
            )

        return coords

    def calculate_angle(self, p1, p2, p3):
        """
        Berechnet den Winkel zwischen drei Punkten.

        Args:
            p1: Erster Punkt (x, y, _)
            p2: Zweiter Punkt (Mittelpunkt) (x, y, _)
            p3: Dritter Punkt (x, y, _)

        Returns:
            float: Winkel in Grad
        """
        # Vektoren berechnen
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])

        # Winkel berechnen
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def calculate_joint_angles(self, coords):
        """
        Berechnet relevante Gelenkwinkel basierend auf den Landmarken.

        Args:
            coords: Dictionary mit Landmarken-Koordinaten

        Returns:
            dict: Dictionary mit Gelenkwinkeln
        """
        if not coords:
            return None

        angles = {}

        # Kniewinkel
        if all(k in coords for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee_angle'] = self.calculate_angle(
                coords['left_hip'], coords['left_knee'], coords['left_ankle']
            )

        if all(k in coords for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_knee_angle'] = self.calculate_angle(
                coords['right_hip'], coords['right_knee'], coords['right_ankle']
            )

        # Hüftwinkel
        if all(k in coords for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['left_hip_angle'] = self.calculate_angle(
                coords['left_shoulder'], coords['left_hip'], coords['left_knee']
            )

        if all(k in coords for k in ['right_shoulder', 'right_hip', 'right_knee']):
            angles['right_hip_angle'] = self.calculate_angle(
                coords['right_shoulder'], coords['right_hip'], coords['right_knee']
            )

        # Rückenwinkel (zwischen Schultern und Hüfte)
        if all(k in coords for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            shoulder_mid = (
                (coords['left_shoulder'][0] + coords['right_shoulder'][0]) // 2,
                (coords['left_shoulder'][1] + coords['right_shoulder'][1]) // 2,
                0
            )
            hip_mid = (
                (coords['left_hip'][0] + coords['right_hip'][0]) // 2,
                (coords['left_hip'][1] + coords['right_hip'][1]) // 2,
                0
            )
            # Vertikale Linie berechnen
            vertical = (shoulder_mid[0], 0, 0)
            angles['back_angle'] = self.calculate_angle(vertical, shoulder_mid, hip_mid)

            # Wenn der Winkel > 90, dann korrigieren
            if angles['back_angle'] > 90:
                angles['back_angle'] = 180 - angles['back_angle']

        return angles

    def release(self):
        """Ressourcen freigeben"""
        self.pose.close()
