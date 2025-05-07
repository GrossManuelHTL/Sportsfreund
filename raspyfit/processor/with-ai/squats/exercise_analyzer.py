import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pose_extractor import PoseExtractor


class ExerciseAnalyzer:
    def __init__(self, model_path, error_categories, sequence_length=30):
        """
        Initialisiert den ExerciseAnalyzer für die Analyse von Übungsvideos.

        Args:
            model_path: Pfad zum trainierten Modell
            error_categories: Liste der Fehlerkategorien
            sequence_length: Anzahl der Frames pro Übungssequenz
        """
        self.model_path = model_path
        self.error_categories = error_categories
        self.sequence_length = sequence_length

        # Initialisieren des PoseExtractors
        self.pose_extractor = PoseExtractor(sequence_length=sequence_length)

        # Prüfen, ob das Modell existiert
        if not os.path.exists(model_path):
            print(f"Warnung: Modell unter {model_path} nicht gefunden!")

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

        if len(landmarks_sequence) != self.sequence_length:
            print(f"Warnung: Unerwartete Sequenzlänge {len(landmarks_sequence)}")
            return None

        # Modell laden und Vorhersage treffen
        model = load_model(self.model_path)
        prediction = model.predict(np.expand_dims(landmarks_sequence, axis=0))[0]

        # Ergebnisse interpretieren
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        feedback = {
            "predicted_category": self.error_categories[predicted_class],
            "confidence": float(confidence),
            "all_probabilities": {cat: float(prob) for cat, prob in zip(self.error_categories, prediction)}
        }

        # Detailliertes Feedback basierend auf der Vorhersage
        feedback["text"] = self.generate_feedback_text(self.error_categories[predicted_class])

        # Visualisierung, falls gewünscht
        if show_visualization and frames:
            self.show_visualization(frames)

        return feedback

    def generate_feedback_text(self, category):
        """
        Generiert detailliertes Feedback basierend auf der erkannten Kategorie.

        Args:
            category: Erkannte Fehlerkategorie

        Returns:
            Feedback-Text
        """
        # Feedback-Texte für Squats
        feedback_texts = {
            "correct": "Ausgezeichnete Squat-Form! Die Haltung ist korrekt.",
            "knees_too_far_apart": "Deine Knie fallen nach außen. Achte darauf, sie in Linie mit deinen Füßen zu halten.",
            "back_not_straight": "Dein Rücken ist nicht gerade. Halte den Rücken während der gesamten Bewegung neutral.",
            "too_high": "Du gehst nicht tief genug. Versuche, deine Oberschenkel parallel zum Boden zu bekommen.",
            "wrong_going_up": "Achte auf korrekte Haltung beim nach oben gehen."
        }

        # Standardtext, falls keine spezifische Kategorie gefunden
        return feedback_texts.get(category, "Keine spezifische Analyse verfügbar für diese Kategorie.")

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