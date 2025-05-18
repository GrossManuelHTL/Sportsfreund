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
        self.model = None

        # Initialisieren des PoseExtractors
        self.pose_extractor = PoseExtractor(sequence_length=sequence_length)

        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}")
                self.model = load_model(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
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
        if self.model is None:
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
        prediction = self.model.predict(np.expand_dims(landmarks_sequence, axis=0))[0]

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

    def predict_from_landmarks(self, landmarks_array):
        """
        Provides form feedback based on direct analysis of landmark positions
        without using the trained model.

        Args:
            landmarks_array: numpy array of shape (33, 3) containing pose landmarks

        Returns:
            Dictionary with analysis results or None if analysis failed
        """
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose

            # Create landmark objects for easier access
            class LandmarkAccess:
                def __init__(self, array):
                    self.data = array

                @property
                def x(self):
                    return self.data[0]

                @property
                def y(self):
                    return self.data[1]

                @property
                def z(self):
                    return self.data[2]

            landmarks = [LandmarkAccess(landmark) for landmark in landmarks_array]

            # Define key landmark indices
            LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
            LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
            LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
            RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
            RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
            RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

            # Calculate angles
            def calculate_angle(a, b, c):
                a = np.array([a.x, a.y])
                b = np.array([b.x, b.y])
                c = np.array([c.x, c.y])

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle

                return angle

            # Calculate key angles
            left_knee_angle = calculate_angle(
                landmarks[LEFT_HIP],
                landmarks[LEFT_KNEE],
                landmarks[LEFT_ANKLE]
            )

            right_knee_angle = calculate_angle(
                landmarks[RIGHT_HIP],
                landmarks[RIGHT_KNEE],
                landmarks[RIGHT_ANKLE]
            )

            left_back_angle = calculate_angle(
                landmarks[LEFT_SHOULDER],
                landmarks[LEFT_HIP],
                landmarks[LEFT_KNEE]
            )

            # Rule-based analysis
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

            # Simple rules for squat form
            category = "correct"
            confidence = 0.7

            if avg_knee_angle > 140:  # Not deep enough
                category = "too_high"
                confidence = 0.8
            elif abs(left_knee_angle - right_knee_angle) > 20:  # Asymmetrical squat
                category = "knees_too_far_apart"
                confidence = 0.7
            elif left_back_angle < 150:  # Back not straight
                category = "back_not_straight"
                confidence = 0.8

            feedback_text = self.generate_feedback_text(category)

            # Create probabilities dict
            probabilities = {cat: 0.1 for cat in self.error_categories}
            probabilities[category] = confidence

            return {
                'predicted_category': category,
                'confidence': confidence,
                'text': feedback_text,
                'all_probabilities': probabilities
            }

        except Exception as e:
            print(f"Error during landmark analysis: {e}")
            return None