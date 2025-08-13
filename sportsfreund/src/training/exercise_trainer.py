"""
Training System for Exercise Models
Handles data preparation and model training for any exercise type
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..core.video_analyzer import VideoAnalyzer
from ..ai.ml_models import MLRepDetectionModel, MLFormValidationModel

class ExerciseTrainer:
    """Trains AI models for exercise analysis"""

    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.video_analyzer = VideoAnalyzer(frame_skip=3)
        self.rep_model = MLRepDetectionModel()
        self.form_model = MLFormValidationModel()

    def train_from_config(self, config_path: str) -> Dict[str, Any]:
        """
        Train models based on configuration file

        Config format:
        {
            "exercise_name": "squat",
            "training_data": {
                "correct_reps": ["path1.mp4", "path2.mp4"],
                "incorrect_form": {
                    "knee_cave": ["bad1.mp4"],
                    "forward_lean": ["bad2.mp4"]
                }
            },
            "rep_detection": {
                "enabled": true,
                "method": "ml"
            },
            "form_validation": {
                "enabled": true,
                "error_types": ["knee_cave", "forward_lean", "shallow_squat"]
            }
        }
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        training_results = {}

        # Prepare training data
        print(f"Preparing training data for {self.exercise_name}...")
        rep_data, form_data = self._prepare_training_data(config['training_data'])

        # Train rep detection model
        if config.get('rep_detection', {}).get('enabled', True):
            print("Training rep detection model...")
            rep_results = self._train_rep_model(rep_data)
            training_results['rep_detection'] = rep_results

            # Save model
            model_dir = Path(f"models/{self.exercise_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            self.rep_model.save_model(str(model_dir / "rep_model.pkl"))

        # Train form validation model
        if config.get('form_validation', {}).get('enabled', True):
            print("Training form validation model...")
            form_results = self._train_form_model(form_data)
            training_results['form_validation'] = form_results

            # Save model
            model_dir = Path(f"models/{self.exercise_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            self.form_model.save_model(str(model_dir / "form_model.pkl"))

        # Save training metadata
        self._save_training_metadata(config, training_results)

        return training_results

    def _prepare_training_data(self, training_config: Dict) -> tuple:
        """Prepare training data from video files"""
        rep_training_data = []
        rep_labels = []
        form_training_data = []
        form_labels = []

        # Process correct rep videos
        correct_videos = training_config.get('correct_reps', [])
        for video_path in correct_videos:
            print(f"Processing correct reps: {video_path}")
            video_data = self.video_analyzer.analyze_video(video_path)

            if video_data['pose_sequence']:
                # Label rep detection data
                rep_frames, rep_frame_labels = self._label_rep_frames(video_data['pose_sequence'])
                rep_training_data.extend(rep_frames)
                rep_labels.extend(rep_frame_labels)

                # Label form data as correct
                for pose_data in video_data['pose_sequence']:
                    form_training_data.append(pose_data)
                    form_labels.append('correct')

        # Process incorrect form videos
        incorrect_form = training_config.get('incorrect_form', {})
        for error_type, video_paths in incorrect_form.items():
            for video_path in video_paths:
                print(f"Processing {error_type} examples: {video_path}")
                video_data = self.video_analyzer.analyze_video(video_path)

                if video_data['pose_sequence']:
                    # Add to form training data
                    for pose_data in video_data['pose_sequence']:
                        form_training_data.append(pose_data)
                        form_labels.append(error_type)

        return (rep_training_data, rep_labels), (form_training_data, form_labels)

    def _label_rep_frames(self, pose_sequence: List[Dict]) -> tuple:
        """Label frames for rep detection using signal processing"""
        from ..ai.ml_models import SignalProcessingRepDetector

        # Extract movement features
        movement_features = self._extract_movement_features(pose_sequence)

        # Detect rep boundaries
        rep_boundaries = SignalProcessingRepDetector.detect_reps_from_movement(
            movement_features, self.exercise_name
        )

        # Create labels (1 for rep center, 0 for others)
        labels = [0] * len(pose_sequence)
        frames = pose_sequence.copy()

        for start, end, quality in rep_boundaries:
            # Mark center of rep as positive example
            center = (start + end) // 2
            if 0 <= center < len(labels):
                labels[center] = 1

        return frames, labels

    def _extract_movement_features(self, pose_sequence: List[Dict]) -> Dict:
        """Extract movement features for signal processing"""
        positions_series = {'hip_height': [], 'knee_height': []}

        for frame_data in pose_sequence:
            for pos_name in positions_series:
                pos_value = frame_data['positions'].get(pos_name, 0)
                positions_series[pos_name].append(pos_value)

        return {
            'positions': {k: np.array(v) for k, v in positions_series.items()}
        }

    def _train_rep_model(self, rep_data: tuple) -> Dict:
        """Train the rep detection model"""
        training_data, labels = rep_data

        if len(training_data) < 10:
            return {'error': 'Insufficient training data for rep detection'}

        return self.rep_model.train(training_data, labels)

    def _train_form_model(self, form_data: tuple) -> Dict:
        """Train the form validation model"""
        training_data, labels = form_data

        if len(training_data) < 10:
            return {'error': 'Insufficient training data for form validation'}

        return self.form_model.train(training_data, labels)

    def _save_training_metadata(self, config: Dict, results: Dict):
        """Save training metadata and results"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert results to JSON-serializable format
        json_serializable_results = convert_numpy_types(results)
        
        metadata = {
            'exercise_name': self.exercise_name,
            'config': config,
            'training_results': json_serializable_results,
            'model_paths': {
                'rep_model': f"models/{self.exercise_name}/rep_model.pkl",
                'form_model': f"models/{self.exercise_name}/form_model.pkl"
            }
        }

        metadata_path = Path(f"models/{self.exercise_name}/training_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
