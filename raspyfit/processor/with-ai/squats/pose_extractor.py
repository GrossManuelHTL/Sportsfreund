import os
import cv2
import numpy as np
import mediapipe as mp


class PoseExtractor:
    def __init__(self, sequence_length=30):
        """
        Initialisiert den PoseExtractor für die Extraktion von Körperpunkten aus Videos.

        Args:
            sequence_length: Anzahl der Frames, die für jede Übung standardisiert werden
        """
        # MediaPipe-Setup für Pose-Erkennung
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Sequenzlänge (Frames je Übung)
        self.sequence_length = sequence_length

        # Definieren relevanter Landmarken für Kniebeugen
        self.relevant_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

    def extract_pose_from_video(self, video_path, visualize=False):
        """
        Extrahiert Körperpunkte aus einem Video.

        Args:
            video_path: Pfad zum Videofile
            visualize: Ob Visualisierungsframes zurückgegeben werden sollen

        Returns:
            landmarks_sequence: Array mit Landmarken-Sequenzen
            frames: (Optional) Liste von Frames mit visualisierten Landmarken
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Konvertieren zu RGB für MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(frame_rgb)

            if result.pose_landmarks:
                # Extrahieren der relevanten Landmarken
                frame_landmarks = []
                for landmark_id in self.relevant_landmarks:
                    landmark = result.pose_landmarks.landmark[landmark_id]
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                landmarks_sequence.append(frame_landmarks)

                # Zeichnen der Landmarken für Visualisierung
                if visualize:
                    annotated_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        result.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    frames.append(annotated_frame)

        cap.release()

        # Normalisieren der Sequenzlänge auf self.sequence_length
        if landmarks_sequence:
            # Interpolation, wenn weniger oder mehr Frames als benötigt
            if len(landmarks_sequence) != self.sequence_length:
                indices = np.linspace(0, len(landmarks_sequence) - 1, self.sequence_length, dtype=int)
                landmarks_sequence = [landmarks_sequence[i] for i in indices]

        return np.array(landmarks_sequence), frames if visualize else None

    def extract_label_from_filename(self, filename, error_categories):
        """
        Extrahiert Label-Informationen aus dem Dateinamen.

        Args:
            filename: Name der Videodatei
            error_categories: Liste mit möglichen Fehlerkategorien

        Returns:
            One-Hot encoded Label
        """
        for category in error_categories:
            if category in filename:
                # One-hot encoding
                label = [0] * len(error_categories)
                label[error_categories.index(category)] = 1
                return label

        # Fallback, falls keine Kategorie erkannt wird
        return [0] * len(error_categories)

    def load_training_data(self, video_dir, error_categories):
        """
        Lädt alle Trainingsvideos aus einem Verzeichnis.

        Args:
            video_dir: Verzeichnis mit Trainingsvideos
            error_categories: Liste mit möglichen Fehlerkategorien

        Returns:
            X: Trainingsfeatures (Landmarken-Sequenzen)
            y: Labels (One-Hot encoded)
            filenames: Liste der Dateinamen
        """
        X = []
        y = []
        filenames = []

        for filename in os.listdir(video_dir):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, filename)
                print(f"Verarbeite {filename}...")
                landmarks_sequence, _ = self.extract_pose_from_video(video_path)

                if len(landmarks_sequence) == self.sequence_length:
                    X.append(landmarks_sequence)
                    y.append(self.extract_label_from_filename(filename, error_categories))
                    filenames.append(filename)
                else:
                    print(f"Überspringe {filename}: Unerwartete Sequenzlänge {len(landmarks_sequence)}")

        return np.array(X), np.array(y), filenames