import cv2
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from pose_extractor import PoseExtractor

class ModelTrainer:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.pose_extractor = PoseExtractor()
        self.models = {}
        self.scalers = {}

    def train_from_videos(self, single_rep_videos, multi_rep_videos):
        print(f"Training {self.exercise_name} phase model")

        all_features = []
        all_phase_labels = []

        for video in single_rep_videos:
            print(f"Processing single rep video: {video}")
            features = self._extract_features(video)
            if features:
                phase_labels = self._label_single_rep_phases(features)
                all_features.extend(features)
                all_phase_labels.extend(phase_labels)
                print(f"Added {len(features)} frames with phases: {set(phase_labels)}")

        for video in multi_rep_videos:
            print(f"Processing multi rep video: {video}")
            features = self._extract_features(video)
            if features:
                phase_labels = self._label_multi_rep_phases(features)
                all_features.extend(features)
                all_phase_labels.extend(phase_labels)
                print(f"Added {len(features)} frames with phases: {set(phase_labels)}")

        if len(all_features) > 20:
            print(f"Training with {len(all_features)} total frames")
            print(f"Phase distribution: {dict(zip(*np.unique(all_phase_labels, return_counts=True)))}")

            self._train_phase_model(all_features, all_phase_labels)
            self._save_models()
            return True
        else:
            print(f"Not enough training data: {len(all_features)} frames")
            return False

    def _extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pose_data = self.pose_extractor.extract_from_frame(frame)
            if pose_data is not None:
                # Extrahiere alle Features wie im RepCounter
                feature_vector = []
                feature_vector.extend(pose_data['raw_landmarks'])
                feature_vector.extend(list(pose_data['angles'].values()))
                feature_vector.extend(list(pose_data['positions'].values()))
                features.append(feature_vector)

        cap.release()
        return features

    def _label_single_rep_phases(self, features):
        """Labelt die Phasen für ein Video mit einer einzelnen Wiederholung"""
        if len(features) < 10:
            return ["unknown"] * len(features)

        # Verwende Kniehöhe als Hauptindikator für Squat-Phasen
        if self.exercise_name == "squat":
            return self._label_squat_phases(features)
        elif self.exercise_name == "pushup":
            return self._label_pushup_phases(features)
        else:
            return self._label_generic_phases(features)

    def _label_squat_phases(self, features):
        """Spezifische Phasen-Labeling für Squats"""
        # Extrahiere Kniehöhe (Durchschnitt beider Knie)
        knee_heights = []
        for f in features:
            left_knee_y = f[75]  # linkes Knie Y
            right_knee_y = f[78]  # rechtes Knie Y
            avg_knee_y = (left_knee_y + right_knee_y) / 2
            knee_heights.append(avg_knee_y)

        knee_heights = np.array(knee_heights)

        # Finde den tiefsten Punkt (höchster Y-Wert)
        bottom_idx = np.argmax(knee_heights)
        total_frames = len(features)

        labels = []
        for i in range(total_frames):
            progress = i / total_frames

            if i <= bottom_idx * 0.3:
                labels.append("start")
            elif i <= bottom_idx * 0.8:
                labels.append("down")
            elif i <= bottom_idx * 1.2:
                labels.append("bottom")
            else:
                labels.append("up")

        return labels

    def _label_pushup_phases(self, features):
        """Spezifische Phasen-Labeling für Push-ups"""
        # Verwende Ellbogen-Y-Position als Indikator
        elbow_heights = []
        for f in features:
            left_elbow_y = f[42]  # linkes Ellbogen Y
            right_elbow_y = f[45]  # rechtes Ellbogen Y
            avg_elbow_y = (left_elbow_y + right_elbow_y) / 2
            elbow_heights.append(avg_elbow_y)

        elbow_heights = np.array(elbow_heights)
        bottom_idx = np.argmax(elbow_heights)
        total_frames = len(features)

        labels = []
        for i in range(total_frames):
            if i <= bottom_idx * 0.3:
                labels.append("start")
            elif i <= bottom_idx * 0.8:
                labels.append("down")
            elif i <= bottom_idx * 1.2:
                labels.append("bottom")
            else:
                labels.append("up")

        return labels

    def _label_generic_phases(self, features):
        """Generisches Phasen-Labeling für unbekannte Übungen"""
        total_frames = len(features)
        quarter = total_frames // 4

        labels = []
        for i in range(total_frames):
            if i < quarter:
                labels.append("start")
            elif i < quarter * 2:
                labels.append("down")
            elif i < quarter * 3:
                labels.append("bottom")
            else:
                labels.append("up")

        return labels

    def _label_multi_rep_phases(self, features):
        """Labelt Phasen für Videos mit mehreren Wiederholungen"""
        if len(features) < 20:
            return ["unknown"] * len(features)

        if self.exercise_name == "squat":
            position_data = [(f[75] + f[78]) / 2 for f in features]  # Kniehöhe
        elif self.exercise_name == "pushup":
            position_data = [(f[42] + f[45]) / 2 for f in features]  # Ellbogenhöhe
        else:
            position_data = [(f[75] + f[78]) / 2 for f in features]  # Default: Kniehöhe

        # Glätte die Daten
        smoothed = np.convolve(position_data, np.ones(5)/5, mode='same')

        # Finde Peaks (höchste Punkte = "start"/"up") und Valleys (tiefste Punkte = "bottom")
        peaks, _ = find_peaks(smoothed, distance=15, prominence=0.02)
        valleys, _ = find_peaks(-smoothed, distance=15, prominence=0.02)

        labels = ["transition"] * len(features)

        # Markiere Bereiche um Peaks als "start" oder "up"
        for peak in peaks:
            for i in range(max(0, peak-5), min(len(labels), peak+6)):
                labels[i] = "start" if i == peak else "up"

        # Markiere Bereiche um Valleys als "bottom"
        for valley in valleys:
            for i in range(max(0, valley-3), min(len(labels), valley+4)):
                labels[i] = "bottom"

        # Markiere Bereiche zwischen start/up und bottom als "down"
        for i in range(len(labels)):
            if labels[i] == "transition":
                # Schaue nach vorherigem und nächstem Label
                prev_significant = self._find_previous_significant_label(labels, i)
                next_significant = self._find_next_significant_label(labels, i)

                if (prev_significant in ["start", "up"] and next_significant == "bottom") or \
                   (prev_significant == "bottom" and next_significant in ["start", "up"]):
                    labels[i] = "down" if prev_significant in ["start", "up"] else "up"

        # Bereinige übrig gebliebene "transition" Labels
        for i in range(len(labels)):
            if labels[i] == "transition":
                labels[i] = "start"  # Default fallback

        return labels

    def _find_previous_significant_label(self, labels, index):
        """Findet das vorherige signifikante Label (nicht 'transition')"""
        for i in range(index-1, -1, -1):
            if labels[i] != "transition":
                return labels[i]
        return "start"

    def _find_next_significant_label(self, labels, index):
        """Findet das nächste signifikante Label (nicht 'transition')"""
        for i in range(index+1, len(labels)):
            if labels[i] != "transition":
                return labels[i]
        return "start"

    def _train_phase_model(self, features, phase_labels):
        """Trainiert das Phasen-Erkennungsmodell"""
        X = np.array(features)
        y = np.array(phase_labels)

        # Entferne 'unknown' Labels für besseres Training
        valid_indices = y != "unknown"
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) < 10:
            print("Not enough valid training data for phase model")
            return

        # Skaliere Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Trainiere Modell mit class_weight für unausgewogene Daten
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        model.fit(X_scaled, y)

        self.models['phase'] = model
        self.scalers['phase'] = scaler

        # Zeige Trainingsstatistiken
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Phase model trained with:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} samples")

        # Evaluiere das Modell
        score = model.score(X_scaled, y)
        print(f"Phase model training accuracy: {score:.3f}")

    def load_models(self):
        """Lädt die trainierten Modelle"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            return False

        phase_model_path = f"{model_dir}/{self.exercise_name}_phase_model.pkl"
        phase_scaler_path = f"{model_dir}/{self.exercise_name}_phase_scaler.pkl"

        try:
            if os.path.exists(phase_model_path) and os.path.exists(phase_scaler_path):
                with open(phase_model_path, 'rb') as f:
                    self.models['phase'] = pickle.load(f)
                with open(phase_scaler_path, 'rb') as f:
                    self.scalers['phase'] = pickle.load(f)
                print(f"Loaded phase model for {self.exercise_name}")
                return True
        except Exception as e:
            print(f"Error loading models: {e}")

        return False

    def _save_models(self):
        """Speichert die trainierten Modelle"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)


        if 'phase' in self.models:
            phase_model_path = f"{model_dir}/{self.exercise_name}_phase_model.pkl"
            phase_scaler_path = f"{model_dir}/{self.exercise_name}_phase_scaler.pkl"

            with open(phase_model_path, 'wb') as f:
                pickle.dump(self.models['phase'], f)
            with open(phase_scaler_path, 'wb') as f:
                pickle.dump(self.scalers['phase'], f)

            print(f"Saved phase model for {self.exercise_name}")

    def predict_phase(self, features):
        """Vorhersage der Phase basierend auf Features"""
        if 'phase' not in self.models or 'phase' not in self.scalers:
            return "unknown"

        try:
            features_scaled = self.scalers['phase'].transform([features])
            phase = self.models['phase'].predict(features_scaled)[0]
            return phase
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown"
