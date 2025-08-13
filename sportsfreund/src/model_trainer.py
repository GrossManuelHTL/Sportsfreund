import cv2
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from pose_extractor import PoseExtractor

class ModelTrainer:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.pose_extractor = PoseExtractor()
        self.models = {}
        self.scalers = {}

    def train_from_videos(self, single_rep_videos, multi_rep_videos):
        print(f"Training {self.exercise_name}")

        all_features = []
        all_labels = []

        for video in single_rep_videos:
            features = self._extract_features(video)
            if features:
                labels = self._label_single_rep(features)
                all_features.extend(features)
                all_labels.extend(labels)

        for video in multi_rep_videos:
            features = self._extract_features(video)
            if features:
                labels = self._label_multi_rep(features)
                all_features.extend(features)
                all_labels.extend(labels)

        if len(all_features) > 20:
            self._train_phase_model(all_features, all_labels)
            self._train_rep_model(all_features, all_labels)
            self._save_models()
            return True
        return False

    def _extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.pose_extractor.extract(frame)
            if landmarks is not None:
                features.append(landmarks)

        cap.release()
        return features

    def _label_single_rep(self, features):
        if len(features) < 10:
            return ["unknown"] * len(features)

        knee_pos = [(f[75] + f[78]) / 2 for f in features]  # avg knee y
        min_idx = np.argmin(knee_pos)
        total = len(features)

        labels = []
        for i in range(total):
            if i < min_idx * 0.4:
                labels.append("start")
            elif i < min_idx * 1.6:
                labels.append("down")
            else:
                labels.append("up")
        return labels

    def _label_multi_rep(self, features):
        if len(features) < 20:
            return ["unknown"] * len(features)

        knee_pos = [(f[75] + f[78]) / 2 for f in features]
        smoothed = np.convolve(knee_pos, np.ones(5)/5, mode='same')

        peaks, _ = find_peaks(smoothed, distance=10)
        valleys, _ = find_peaks(-smoothed, distance=10)

        labels = ["transition"] * len(features)

        for peak in peaks:
            for i in range(max(0, peak-3), min(len(labels), peak+4)):
                labels[i] = "up"

        for valley in valleys:
            for i in range(max(0, valley-3), min(len(labels), valley+4)):
                labels[i] = "down"

        return labels

    def _train_phase_model(self, features, labels):
        X = np.array(features)
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)

        self.models['phase'] = model
        self.scalers['phase'] = scaler

    def _train_rep_model(self, features, labels):
        rep_features = []
        rep_labels = []
        
        print(f"Training rep model with {len(features)} features and labels: {set(labels)}")
        
        # Neue Strategie: Lerne das komplette Rep-Pattern aus Single-Rep Videos
        # Suche nach vollständigen Bewegungszyklen: start -> down -> up
        
        i = 0
        while i < len(labels) - 10:  # Mindestens 10 Frames für einen Rep
            # Suche nach Start eines Reps
            if labels[i] in ["start", "up"]:
                # Schaue voraus ob ein kompletter Zyklus folgt
                cycle_end = self._find_complete_cycle(labels, i)
                
                if cycle_end > i:
                    # Kompletter Zyklus gefunden - markiere Mittelpunkt als Rep
                    cycle_length = cycle_end - i
                    rep_center = i + cycle_length // 2
                    
                    # Der zentrale Frame ist ein Rep
                    rep_features.append(features[rep_center])
                    rep_labels.append("rep")
                    
                    # Frames drumherum sind keine Reps
                    for j in range(max(0, i), min(len(features), cycle_end)):
                        if j != rep_center:
                            rep_features.append(features[j])
                            rep_labels.append("no_rep")
                    
                    i = cycle_end  # Springe zum Ende des Zyklus
                else:
                    # Kein kompletter Zyklus - markiere als no_rep
                    rep_features.append(features[i])
                    rep_labels.append("no_rep")
                    i += 1
            else:
                # Übergangsframe - markiere als no_rep
                rep_features.append(features[i])
                rep_labels.append("no_rep")
                i += 1
        
        print(f"Rep model data: {len(rep_features)} samples, {rep_labels.count('rep')} reps, {rep_labels.count('no_rep')} no_reps")
        
        if len(rep_features) > 10 and rep_labels.count('rep') > 0 and rep_labels.count('no_rep') > 0:
            X = np.array(rep_features)
            y = np.array(rep_labels)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
            model.fit(X_scaled, y)
            
            self.models['rep'] = model
            self.scalers['rep'] = scaler
            print("Rep model trained successfully!")
            print(f"Model learned from {rep_labels.count('rep')} rep examples")
        else:
            print(f"Not enough data for rep model: {len(rep_features)} samples, {rep_labels.count('rep')} reps")
    
    def _find_complete_cycle(self, labels, start_idx):
        """Findet das Ende eines kompletten Bewegungszyklus"""
        down_seen = False
        up_seen = False
        
        for i in range(start_idx, min(len(labels), start_idx + 30)):  # Max 30 Frames pro Rep
            if labels[i] == "down":
                down_seen = True
            elif down_seen and labels[i] in ["up", "start"]:
                up_seen = True
                return i + 1  # Ende des Zyklus
        
        return start_idx  # Kein kompletter Zyklus gefunden

    def _save_models(self):
        os.makedirs("models", exist_ok=True)

        data = {
            'exercise': self.exercise_name,
            'models': self.models,
            'scalers': self.scalers
        }

        with open(f"models/{self.exercise_name}.pkl", 'wb') as f:
            pickle.dump(data, f)

    def load_models(self):
        try:
            with open(f"models/{self.exercise_name}.pkl", 'rb') as f:
                data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']
            return True
        except:
            return False
