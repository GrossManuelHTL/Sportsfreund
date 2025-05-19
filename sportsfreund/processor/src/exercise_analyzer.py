import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pose_extractor import PoseExtractor
from exercise_manager import ExerciseManager


class ExerciseAnalyzer:
    def __init__(self, exercise_name):
        """
        Initializes the ExerciseAnalyzer for analyzing videos of an Exercise.

        Args:
            exercise_config: path to the exercise configuration file or a dictionary containing the exercise configuration
        """

        self.manager = ExerciseManager()

        exercise_config = self.manager.get_exercise_config(exercise_name)

        self.pose_extractor = PoseExtractor(exercise_config)

        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, self.pose_extractor.get_model_name())

        # Pfad für das Phasenerkennungsmodell
        base_name, ext = os.path.splitext(self.pose_extractor.get_model_name())
        self.phase_model_path = os.path.join(self.model_dir, f"{base_name}_phases{ext}")

        if not os.path.exists(self.model_path):
            print(f"Warnung: Modell unter {self.model_path} nicht gefunden!")

        # Wiederholungserkennung
        self.rep_config = None
        if "rep_detection" in self.pose_extractor.config:
            self.rep_config = self.pose_extractor.config["rep_detection"]

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

        if len(landmarks_sequence) == 0:
            print(f"Warnung: Keine Landmarken im Video erkannt")
            return None

        model = load_model(self.model_path)

        # Standard-Analyse der Übungsqualität für das Gesamtvideo
        if len(landmarks_sequence) == self.pose_extractor.sequence_length:
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

            # Wenn auch Wiederholungen erkannt werden sollen
            if os.path.exists(self.phase_model_path) and self.rep_config:
                rep_results = self.analyze_repetitions(video_path)
                feedback["repetitions"] = rep_results

        else:
            print(f"Warnung: Unerwartete Sequenzlänge {len(landmarks_sequence)}")

            # Wenn auch Wiederholungen erkannt werden sollen
            if os.path.exists(self.phase_model_path) and self.rep_config:
                feedback = {
                    "exercise": self.pose_extractor.get_exercise_name(),
                    "text": "Video hat unerwartete Länge für Übungsqualitätsanalyse, aber Wiederholungen wurden erkannt."
                }
                rep_results = self.analyze_repetitions(video_path)
                feedback["repetitions"] = rep_results
            else:
                return None

        if show_visualization and frames:
            self.show_visualization(frames)

        return feedback

    def analyze_repetitions(self, video_path):
        """
        Analysiert ein Video auf Übungswiederholungen und gibt individuelles Feedback für jede Wiederholung.

        Args:
            video_path: Pfad zum Analysevideo

        Returns:
            Liste mit Informationen zu jeder erkannten Wiederholung
        """
        print("Analysiere Wiederholungen...")

        # Laden des Phasenmodells
        phase_model = load_model(self.phase_model_path)
        phases = self.pose_extractor.get_phases()

        # Wiederholungsparameter aus der Konfiguration
        completed_sequence = self.rep_config.get("completed_sequence", [])
        count_on_phase = self.rep_config.get("count_on_phase", phases[0] if phases else None)
        min_phase_duration = self.rep_config.get("min_phase_duration", 5)

        # Pose-Landmarken extrahieren (ohne Normalisierung der Sequenzlänge)
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose_extractor.pose.process(frame_rgb)

            if result.pose_landmarks:
                frame_landmarks = []
                for landmark_id in self.pose_extractor.relevant_landmarks:
                    landmark = result.pose_landmarks.landmark[landmark_id]
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                landmarks_sequence.append(frame_landmarks)
                frames.append(frame)

        cap.release()

        if not landmarks_sequence:
            return {
                "count": 0,
                "message": "Keine Landmarken erkannt"
            }

        # Phasen für jeden Frame vorhersagen
        sliding_window_size = self.pose_extractor.sequence_length
        all_phases = []

        for i in range(0, len(landmarks_sequence) - sliding_window_size + 1, 5):  # Schritt 5 für Geschwindigkeit
            window = np.array(landmarks_sequence[i:i+sliding_window_size])
            prediction = phase_model.predict(np.expand_dims(window, axis=0))[0]
            predicted_phase_idx = np.argmax(prediction)
            all_phases.append({
                "frame_idx": i + sliding_window_size // 2,  # Mittlerer Frame des Fensters
                "phase": phases[predicted_phase_idx],
                "confidence": float(prediction[predicted_phase_idx])
            })

        # Glätten der Phasen mit einem einfachen Mehrheitsvoting
        smoothed_phases = []
        window_size = 5

        for i in range(len(all_phases)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(all_phases) - 1, i + window_size // 2)
            window_phases = [all_phases[j]["phase"] for j in range(start_idx, end_idx + 1)]

            # Mehrheitsvoting
            phase_counts = {}
            for phase in window_phases:
                if phase in phase_counts:
                    phase_counts[phase] += 1
                else:
                    phase_counts[phase] = 1

            max_count = 0
            majority_phase = all_phases[i]["phase"]
            for phase, count in phase_counts.items():
                if count > max_count:
                    max_count = count
                    majority_phase = phase

            print(f"Frame {all_phases[i]['frame_idx']}: Phase {majority_phase} mit {max_count} Stimmen")

            smoothed_phases.append({
                "frame_idx": all_phases[i]["frame_idx"],
                "phase": majority_phase,
                "confidence": all_phases[i]["confidence"]
            })

        # Zusammenhängende Segmente von Phasen finden
        phase_segments = []
        current_segment = None

        for phase_info in smoothed_phases:
            if current_segment is None or current_segment["phase"] != phase_info["phase"]:
                if current_segment is not None:
                    current_segment["end_frame"] = phase_info["frame_idx"] - 1
                    phase_segments.append(current_segment)

                current_segment = {
                    "phase": phase_info["phase"],
                    "start_frame": phase_info["frame_idx"],
                    "end_frame": phase_info["frame_idx"],
                    "confidence": phase_info["confidence"]
                }
            else:
                current_segment["end_frame"] = phase_info["frame_idx"]
                current_segment["confidence"] = (current_segment["confidence"] + phase_info["confidence"]) / 2

        if current_segment is not None:
            phase_segments.append(current_segment)

        # Filtern von zu kurzen Segmenten
        filtered_segments = [segment for segment in phase_segments
                             if segment["end_frame"] - segment["start_frame"] >= min_phase_duration]

        # Wiederholungen zählen
        repetitions = []
        sequence_idx = 0
        current_rep = {"phases": []}

        for segment in filtered_segments:
            if sequence_idx < len(completed_sequence) and segment["phase"] == completed_sequence[sequence_idx]:
                current_rep["phases"].append({
                    "phase": segment["phase"],
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"]
                })

                if segment["phase"] == count_on_phase:
                    # Start- und Endframe der gesamten Wiederholung ermitteln
                    if not "start_frame" in current_rep:
                        current_rep["start_frame"] = segment["start_frame"]
                    current_rep["end_frame"] = segment["end_frame"]

                sequence_idx += 1

                # Wenn die Sequenz komplett ist, Wiederholung speichern und zurücksetzen
                if sequence_idx == len(completed_sequence):
                    # Wenn die Wiederholung vollständig ist, für diese ein Feedback generieren
                    rep_feedback = self.generate_rep_feedback(landmarks_sequence,
                                                             current_rep["start_frame"],
                                                             current_rep["end_frame"])
                    current_rep["feedback"] = rep_feedback
                    repetitions.append(current_rep)

                    # Zurücksetzen für die nächste Wiederholung
                    sequence_idx = 0
                    current_rep = {"phases": []}
            else:
                # Wenn die Phase nicht der erwarteten Phase in der Sequenz entspricht,
                # zurücksetzen und prüfen, ob es die erste Phase einer neuen Sequenz ist
                if segment["phase"] == completed_sequence[0]:
                    sequence_idx = 1
                    current_rep = {
                        "phases": [{
                            "phase": segment["phase"],
                            "start_frame": segment["start_frame"],
                            "end_frame": segment["end_frame"]
                        }]
                    }

                    if segment["phase"] == count_on_phase:
                        current_rep["start_frame"] = segment["start_frame"]
                        current_rep["end_frame"] = segment["end_frame"]
                else:
                    sequence_idx = 0
                    current_rep = {"phases": []}

        # Zusammenfassung der erkannten Wiederholungen
        repetition_summary = {
            "count": len(repetitions),
            "repetitions": repetitions
        }

        return repetition_summary

    def generate_rep_feedback(self, landmarks_sequence, start_frame, end_frame):
        """
        Generiert Feedback für eine spezifische Wiederholung

        Args:
            landmarks_sequence: Sequenz der Landmarken des gesamten Videos
            start_frame: Startframe der Wiederholung
            end_frame: Endframe der Wiederholung

        Returns:
            Feedback für die Wiederholung
        """
        # Überprüfen, ob Landmarken für die gesamte Wiederholung vorhanden sind
        if start_frame >= len(landmarks_sequence) or end_frame >= len(landmarks_sequence):
            return {"text": "Keine ausreichenden Daten für die Analyse dieser Wiederholung"}

        # Extrahieren der Landmarken für die Wiederholung
        rep_landmarks = landmarks_sequence[start_frame:end_frame+1]

        # Wenn zu wenige Landmarken für die Analyse vorhanden sind, abbrechen
        if len(rep_landmarks) < self.pose_extractor.sequence_length:
            # Wenn nötig, durch Interpolation auf die erforderliche Länge bringen
            indices = np.linspace(0, len(rep_landmarks) - 1, self.pose_extractor.sequence_length, dtype=int)
            rep_landmarks = [rep_landmarks[i] for i in indices]

        # Wenn zu viele Landmarken vorhanden sind, durch Sampling auf die richtige Länge bringen
        if len(rep_landmarks) > self.pose_extractor.sequence_length:
            indices = np.linspace(0, len(rep_landmarks) - 1, self.pose_extractor.sequence_length, dtype=int)
            rep_landmarks = [rep_landmarks[i] for i in indices]

        # Normalisieren der Landmarken auf die erwartete Sequenzlänge
        rep_landmarks = np.array(rep_landmarks)

        # Laden des Hauptmodells für Übungsqualität
        model = load_model(self.model_path)

        # Übungsqualität für diese Wiederholung vorhersagen
        prediction = model.predict(np.expand_dims(rep_landmarks, axis=0))[0]
        categories = self.pose_extractor.get_categories()
        predicted_class = np.argmax(prediction)

        # Feedback zusammenstellen
        feedback = {
            "category": categories[predicted_class],
            "confidence": float(prediction[predicted_class]),
            "text": self.generate_feedback_text(categories[predicted_class]),
            "all_probabilities": {cat: float(prob) for cat, prob in zip(categories, prediction)}
        }

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
        for i, frame in enumerate(frames):
            cv2.imshow("Pose Detection", frame)
            key = cv2.waitKey(delay)
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()

