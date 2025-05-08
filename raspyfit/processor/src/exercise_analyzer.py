import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pose_extractor import PoseExtractor


class ExerciseAnalyzer:
    def __init__(self, exercise_config):
        """
        Initialisiert den ExerciseAnalyzer für die Analyse von Übungsvideos.

        Args:
            exercise_config: Pfad zur Exercise-Konfig oder Konfig-Dictionary
        """
        # Pose-Extraktor mit der Übungskonfiguration initialisieren
        self.pose_extractor = PoseExtractor(exercise_config)

        # Modellpfad bestimmen
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, self.pose_extractor.get_model_name())

        # Prüfen, ob das Modell existiert
        if not os.path.exists(self.model_path):
            print(f"Warnung: Modell unter {self.model_path} nicht gefunden!")

    def analyze_video(self, video_path, show_visualization=False):
        """
        Analysiert ein Video und gibt Feedback zur Übungsausführung.

        Args:
            video_path: Pfad zum Übungsvideo
            show_visualization: Ob die Visualisierung angezeigt werden soll

        Returns:
            Dictionary mit Feedback-Informationen oder None bei Fehler
        """
        if not os.path.exists(self.model_path):
            print(f"Fehler: Trainiertes Modell unter {self.model_path} nicht gefunden! Bitte zuerst trainieren.")
            return None

        # Landmarken extrahieren
        print(f"Analysiere Video: {video_path}")
        landmarks_sequence, frames = self.pose_extractor.extract_pose_from_video(video_path,
                                                                                 visualize=show_visualization)

        if len(landmarks_sequence) != self.pose_extractor.sequence_length:
            print(f"Warnung: Unerwartete Sequenzlänge {len(landmarks_sequence)}")
            return None

        # Modell laden und Vorhersage treffen
        model = load_model(self.model_path)
        prediction = model.predict(np.expand_dims(landmarks_sequence, axis=0))[0]

        # Kategorien aus der Konfiguration holen
        categories = self.pose_extractor.get_categories()

        # Ergebnisse interpretieren
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        feedback = {
            "exercise": self.pose_extractor.get_exercise_name(),
            "predicted_category": categories[predicted_class],
            "confidence": float(confidence),
            "all_probabilities": {cat: float(prob) for cat, prob in zip(categories, prediction)}
        }

        # Detailliertes Feedback basierend auf der Vorhersage
        feedback["text"] = self.generate_feedback_text(categories[predicted_class])

        # Visualisierung, falls gewünscht
        if show_visualization and frames:
            self.show_visualization(frames)

        return feedback

    def generate_feedback_text(self, category_id):
        """
        Generiert detailliertes Feedback basierend auf der erkannten Kategorie.

        Args:
            category_id: ID der erkannten Kategorie

        Returns:
            Feedback-Text
        """
        # Feedback aus der Konfiguration holen
        if self.pose_extractor.config and "categories" in self.pose_extractor.config:
            for category in self.pose_extractor.config["categories"]:
                if category["id"] == category_id:
                    return category["feedback"]

        # Standardtext, falls keine spezifische Kategorie gefunden
        return "Keine spezifische Analyse verfügbar für diese Kategorie."

    def show_visualization(self, frames, delay=30):
        """
        Zeigt die Visualisierung der Posen-Erkennung an.

        Args:
            frames: Liste von Frames mit visualisierten Landmarken
            delay: Verzögerung zwischen den Frames in ms
        """
        for frame in frames:
            cv2.imshow('Pose Analysis', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()