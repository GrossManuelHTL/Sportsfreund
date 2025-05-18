import os
import cv2
import numpy as np
import mediapipe as mp
import json


class PoseExtractor:
    def __init__(self, exercise_config=None, sequence_length=30):
        """
        Initialisiert den PoseExtractor für die Extraktion von Körperpunkten.

        Args:
            exercise_config: Pfad zur Exercise-Konfig oder Konfig-Dictionary
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

        # Standardeinstellungen
        self.sequence_length = sequence_length
        self.config = exercise_config
        self.relevant_landmarks = []

        if "relevant_landmarks" in self.config:
            self.relevant_landmarks = [
                getattr(self.mp_pose.PoseLandmark, landmark_name)
                for landmark_name in self.config["relevant_landmarks"]
            ]
        else:
            # Standardmäßig alle Landmarken verwenden
            self.relevant_landmarks = list(range(33))

            if "sequence_length" in self.config:
                self.sequence_length = self.config["sequence_length"]

    def get_categories(self):
        """Gibt die Kategorien aus der Konfiguration zurück"""
        if self.config and "categories" in self.config:
            return [cat["id"] for cat in self.config["categories"]]
        return []

    def get_exercise_name(self):
        """Gibt den Namen der Übung zurück"""
        if self.config and "exercise_name" in self.config:
            return self.config["exercise_name"]
        return "Unbekannte Übung"

    def get_model_name(self):
        """Gibt den Modellnamen aus der Konfiguration zurück"""
        if self.config and "model_name" in self.config:
            return self.config["model_name"]
        return "exercise_model.h5"

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

    def extract_label_from_filename(self, filename):
        """
        Extrahiert Label-Informationen aus dem Dateinamen.

        Args:
            filename: Name der Videodatei

        Returns:
            One-Hot encoded Label
        """
        categories = self.get_categories()

        for category in categories:
            if category in filename:
                # One-hot encoding
                label = [0] * len(categories)
                label[categories.index(category)] = 1
                return label

        # Fallback, falls keine Kategorie erkannt wird
        return [0] * len(categories)

    def load_training_data(self, video_dir):
        """
        Lädt alle Trainingsvideos aus einem Verzeichnis.

        Args:
            video_dir: Verzeichnis mit Trainingsvideos

        Returns:
            X: Trainingsfeatures (Landmarken-Sequenzen)
            y: Labels (One-Hot encoded)
            filenames: Liste der Dateinamen
        """
        X = []
        y = []
        filenames = []

        # Video-Präfix aus Konfiguration
        video_prefix = self.config.get("video_prefix", "")

        for filename in os.listdir(video_dir):
            # Nur Dateien mit dem richtigen Präfix und Videoformat verarbeiten
            if (not video_prefix or filename.startswith(video_prefix)) and \
                    filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, filename)
                print(f"Verarbeite {filename}...")
                landmarks_sequence, _ = self.extract_pose_from_video(video_path)

                if len(landmarks_sequence) == self.sequence_length:
                    X.append(landmarks_sequence)
                    y.append(self.extract_label_from_filename(filename))
                    filenames.append(filename)
                else:
                    print(f"Überspringe {filename}: Unerwartete Sequenzlänge {len(landmarks_sequence)}")

        return np.array(X), np.array(y), filenames