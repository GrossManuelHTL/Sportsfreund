"""
Machine Learning models for rep detection and form validation
"""
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks
from typing import Dict, List, Any
from pathlib import Path
from exercise_base import RepDetectionModel, FormValidationModel

class MLRepDetectionModel(RepDetectionModel):
    """Machine Learning based rep detection using Random Forest"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, training_data: List[Dict], labels: List[int]) -> Dict:
        """
        Train rep detection model

        Args:
            training_data: List of pose feature dictionaries
            labels: Binary labels (1 = rep center, 0 = no rep)
        """
        if len(training_data) != len(labels):
            raise ValueError("Training data and labels must have same length")

        # Extract features for ML
        X = []
        for pose_data in training_data:
            features = self._extract_ml_features(pose_data)
            X.append(features)

        X = np.array(X)
        y = np.array(labels)

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Evaluate on training data (basic check)
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        return {
            'accuracy': accuracy,
            'total_samples': len(X),
            'rep_samples': np.sum(y),
            'feature_importance': dict(zip(
                ['knee_angle_avg', 'hip_height', 'knee_distance', 'movement_velocity'],
                self.model.feature_importances_
            ))
        }

    def predict(self, pose_data: Dict) -> Dict[str, float]:
        """Predict if current pose indicates a rep"""
        if not self.is_trained:
            return {'rep_probability': 0.0, 'confidence': 0.0}

        features = self._extract_ml_features(pose_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of being a rep

        return {
            'rep_probability': float(probability),
            'prediction': int(prediction),
            'confidence': float(max(self.model.predict_proba(features_scaled)[0]))
        }

    def _extract_ml_features(self, pose_data: Dict) -> np.ndarray:
        """Extract ML-ready features from pose data"""
        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})

        features = [
            (angles.get('left_knee', 0) + angles.get('right_knee', 0)) / 2,  # avg knee angle
            positions.get('hip_height', 0),
            positions.get('knee_distance', 0),
            0.0  # movement velocity placeholder (would need frame sequence)
        ]

        return np.array(features)

    def save_model(self, path: str) -> bool:
        """Save model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load model from file"""
        try:
            if not Path(path).exists():
                return False

            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class MLFormValidationModel(FormValidationModel):
    """Machine Learning based form validation"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.error_types = []
        self.is_trained = False

    def train(self, training_data: List[Dict], error_labels: List[str]) -> Dict:
        """
        Train form validation model

        Args:
            training_data: List of pose feature dictionaries
            error_labels: Error type labels ('correct', 'knee_cave', 'forward_lean', etc.)
        """
        if len(training_data) != len(error_labels):
            raise ValueError("Training data and labels must have same length")

        # Extract features
        X = []
        for pose_data in training_data:
            features = self._extract_form_features(pose_data)
            X.append(features)

        X = np.array(X)
        y = np.array(error_labels)

        # Store unique error types
        self.error_types = list(set(error_labels))

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        return {
            'accuracy': accuracy,
            'total_samples': len(X),
            'error_types': self.error_types,
            'class_distribution': {label: list(y).count(label) for label in self.error_types}
        }

    def validate_form(self, pose_data: Dict) -> Dict[str, Any]:
        """Validate form and detect errors"""
        if not self.is_trained:
            return {'error_type': 'unknown', 'confidence': 0.0, 'is_correct': False}

        features = self._extract_form_features(pose_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)

        return {
            'error_type': prediction,
            'confidence': float(confidence),
            'is_correct': prediction == 'correct',
            'error_probabilities': dict(zip(self.model.classes_, probabilities))
        }

    def _extract_form_features(self, pose_data: Dict) -> np.ndarray:
        """Extract form-specific features"""
        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})

        features = [
            angles.get('left_knee', 0),
            angles.get('right_knee', 0),
            angles.get('back', 0),
            abs(angles.get('left_knee', 0) - angles.get('right_knee', 0)),  # knee asymmetry
            positions.get('knee_distance', 0),
            positions.get('hip_height', 0)
        ]

        return np.array(features)

    def save_model(self, path: str) -> bool:
        """Save form validation model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'error_types': self.error_types,
                'is_trained': self.is_trained
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            print(f"Error saving form model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load form validation model"""
        try:
            if not Path(path).exists():
                return False

            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.error_types = model_data['error_types']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading form model: {e}")
            return False

class SignalProcessingRepDetector:
    """Signal processing based rep detection using peak detection"""

    @staticmethod
    def detect_reps_from_movement(movement_features: Dict, exercise_type: str = 'squat') -> List[tuple]:
        """
        Detect rep boundaries using signal processing

        Args:
            movement_features: Time series movement data
            exercise_type: Type of exercise for parameter tuning

        Returns:
            List of (start_frame, end_frame, rep_quality_score) tuples
        """
        if exercise_type == 'squat':
            return SignalProcessingRepDetector._detect_squat_reps(movement_features)
        else:
            # Generic detection
            return SignalProcessingRepDetector._detect_generic_reps(movement_features)

    @staticmethod
    def _detect_squat_reps(movement_features: Dict) -> List[tuple]:
        """Detect squat reps using hip height signal"""
        hip_heights = movement_features['positions'].get('hip_height', np.array([]))

        if len(hip_heights) < 10:
            return []

        # Smooth signal
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hip_heights, sigma=2)

        # Find valleys (bottom of squat)
        valleys, _ = find_peaks(-smoothed, distance=15, prominence=0.02)

        # Find peaks around valleys
        reps = []
        for valley in valleys:
            # Look for start and end of rep
            start = max(0, valley - 15)
            end = min(len(smoothed), valley + 15)

            # Calculate rep quality based on depth
            rep_depth = max(smoothed[start:end]) - min(smoothed[start:end])
            quality_score = min(1.0, rep_depth / 0.1)  # Normalize depth

            reps.append((start, end, quality_score))

        return reps

    @staticmethod
    def _detect_generic_reps(movement_features: Dict) -> List[tuple]:
        """Generic rep detection using primary movement signal"""
        # Use hip height as default primary signal
        signal = movement_features['positions'].get('hip_height', np.array([]))

        if len(signal) < 10:
            return []

        # Simple peak detection
        peaks, _ = find_peaks(signal, distance=10)
        valleys, _ = find_peaks(-signal, distance=10)

        # Pair peaks and valleys to form reps
        reps = []
        for i in range(len(valleys)):
            start = valleys[i] - 5 if i == 0 else (valleys[i-1] + valleys[i]) // 2
            end = valleys[i] + 5 if i == len(valleys)-1 else (valleys[i] + valleys[i+1]) // 2
            reps.append((max(0, start), min(len(signal), end), 1.0))

        return reps
