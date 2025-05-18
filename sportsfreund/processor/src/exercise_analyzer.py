import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pose_extractor import PoseExtractor


class ExerciseAnalyzer:
    def __init__(self, exercise_config):
        """
        Initializes the ExerciseAnalyzer for analyzing videos of an Exercise.

        Args:
            exercise_config: path to the exercise configuration file or a dictionary containing the exercise configuration
        """

        self.pose_extractor = PoseExtractor(exercise_config)

        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, self.pose_extractor.get_model_name())

        if not os.path.exists(self.model_path):
            print(f"Warnung: Modell unter {self.model_path} nicht gefunden!")

    def analyze_video(self, video_path, show_visualization=False):
        """
        analizes a video and returns feedback for the exercise.

        Args:
            video_path: path to the video to be analyzed
            show_visualization: if True, the visualization of the pose detection is shown

        Returns:
            dictionary containing feedback information or None if an error occurred
        """
        if not os.path.exists(self.model_path):
            print(f"Fehler: Trainiertes Modell unter {self.model_path} nicht gefunden! Bitte zuerst trainieren.")
            return None

        print(f"Analysiere Video: {video_path}")
        landmarks_sequence, frames = self.pose_extractor.extract_pose_from_video(video_path,
                                                                                 visualize=show_visualization)

        if len(landmarks_sequence) != self.pose_extractor.sequence_length:
            print(f"Warnung: Unerwartete Sequenzlänge {len(landmarks_sequence)}")
            return None


        model = load_model(self.model_path)
        prediction = model.predict(np.expand_dims(landmarks_sequence, axis=0))[0]

        categories = self.pose_extractor.get_categories()

        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        feedback = {
            "exercise": self.pose_extractor.get_exercise_name(),
            "predicted_category": categories[predicted_class],
            "confidence": float(confidence),
            "all_probabilities": {cat: float(prob) for cat, prob in zip(categories, prediction)}
        }

        feedback["text"] = self.generate_feedback_text(categories[predicted_class])

        if show_visualization and frames:
            self.show_visualization(frames)

        return feedback

    def generate_feedback_text(self, category_id):
        """
        generates feedback text based on the predicted category.

        Args:
            category_id: id of the predicted category

        Returns:
            Feedback-Text
        """
        if self.pose_extractor.config and "categories" in self.pose_extractor.config:
            for category in self.pose_extractor.config["categories"]:
                if category["id"] == category_id:
                    return category["feedback"]

        return "Keine spezifische Analyse verfügbar für diese Kategorie."

    def show_visualization(self, frames, delay=30):
        """
        Shows the visualization of the pose detection.

        Args:
            frames: lists of frames with visualized landmarks
            delay: delay between frames in ms
        """
        for frame in frames:
            cv2.imshow('Pose Analysis', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()