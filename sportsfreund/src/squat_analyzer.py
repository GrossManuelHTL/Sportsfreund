"""
Squat Exercise Analyzer
Implements squat-specific rep detection and form validation
"""
import numpy as np
from typing import Dict, List, Any
from exercise_base import ExerciseAnalyzer
from ml_models import MLRepDetectionModel, MLFormValidationModel, SignalProcessingRepDetector

class SquatAnalyzer(ExerciseAnalyzer):
    """Squat-specific exercise analyzer"""

    def __init__(self):
        super().__init__("squat")
        self.rep_model = MLRepDetectionModel()
        self.form_model = MLFormValidationModel()
        self.min_squat_depth = 90  # degrees
        self.max_knee_valgus = 15  # degrees

    def analyze_rep(self, pose_sequence: List[Dict]) -> Dict[str, Any]:
        """Analyze squat repetitions in pose sequence"""
        if not pose_sequence:
            return {'rep_count': 0, 'errors': [], 'feedback': 'No pose data detected'}

        # Extract movement features
        movement_features = self._extract_movement_features(pose_sequence)

        # Detect rep boundaries using signal processing
        rep_boundaries = self.detect_rep_boundaries(movement_features)

        # Analyze each detected rep
        rep_analyses = []
        total_errors = []

        for start, end, quality in rep_boundaries:
            rep_data = {
                'pose_sequence': pose_sequence[start:end],
                'movement_features': {
                    k: v[start:end] if isinstance(v, np.ndarray) else v
                    for k, v in movement_features.items()
                },
                'quality_score': quality
            }

            rep_analysis = self.validate_rep_form(rep_data)
            rep_analyses.append(rep_analysis)
            total_errors.extend(rep_analysis.get('errors', []))

        return {
            'rep_count': len(rep_boundaries),
            'rep_analyses': rep_analyses,
            'total_errors': total_errors,
            'feedback': self._generate_feedback(rep_analyses, total_errors),
            'average_quality': np.mean([r['quality_score'] for r in rep_analyses]) if rep_analyses else 0
        }

    def detect_rep_boundaries(self, movement_features: Dict) -> List[tuple]:
        """Detect squat rep boundaries using hip height"""
        return SignalProcessingRepDetector.detect_reps_from_movement(movement_features, 'squat')

    def validate_rep_form(self, rep_data: Dict) -> Dict[str, Any]:
        """Validate squat form for a single rep"""
        pose_sequence = rep_data['pose_sequence']
        errors = []
        warnings = []

        if not pose_sequence:
            return {'valid': False, 'errors': ['No pose data'], 'warnings': [], 'score': 0}

        # Analyze deepest point of squat (usually middle of sequence)
        mid_point = len(pose_sequence) // 2
        deepest_pose = pose_sequence[mid_point]

        # Check squat depth
        avg_knee_angle = (deepest_pose['angles']['left_knee'] + deepest_pose['angles']['right_knee']) / 2
        if avg_knee_angle > self.min_squat_depth:
            errors.append('squat_not_deep_enough')

        # Check knee alignment
        knee_asymmetry = abs(deepest_pose['angles']['left_knee'] - deepest_pose['angles']['right_knee'])
        if knee_asymmetry > self.max_knee_valgus:
            errors.append('knee_asymmetry')

        # Check forward lean
        if deepest_pose['angles']['back'] < 160:
            warnings.append('forward_lean')

        # Check knee tracking
        knee_distance = deepest_pose['positions']['knee_distance']
        ankle_distance = deepest_pose['positions']['ankle_distance']
        if knee_distance < ankle_distance * 0.8:
            errors.append('knee_valgus')

        # Use ML model for additional validation if trained
        ml_validation = {}
        if self.form_model.is_trained:
            ml_validation = self.form_model.validate_form(deepest_pose)
            if not ml_validation['is_correct']:
                errors.append(f"ml_detected_{ml_validation['error_type']}")

        # Calculate score
        score = 100
        score -= len(errors) * 25
        score -= len(warnings) * 10
        score = max(0, score)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'score': score,
            'quality_score': rep_data['quality_score'],
            'ml_validation': ml_validation,
            'key_metrics': {
                'depth_angle': avg_knee_angle,
                'knee_asymmetry': knee_asymmetry,
                'back_angle': deepest_pose['angles']['back'],
                'knee_tracking': knee_distance / ankle_distance if ankle_distance > 0 else 0
            }
        }

    def _extract_movement_features(self, pose_sequence: List[Dict]) -> Dict:
        """Extract movement features from pose sequence"""
        angles_series = {
            'left_knee': [],
            'right_knee': [],
            'left_hip': [],
            'right_hip': [],
            'back': []
        }

        positions_series = {
            'hip_height': [],
            'knee_height': [],
            'ankle_height': [],
            'knee_distance': [],
            'ankle_distance': []
        }

        timestamps = []

        for i, frame_data in enumerate(pose_sequence):
            timestamps.append(i)  # Frame index as timestamp

            for angle_name in angles_series:
                angle_value = frame_data['angles'].get(angle_name, 0)
                angles_series[angle_name].append(angle_value)

            for pos_name in positions_series:
                pos_value = frame_data['positions'].get(pos_name, 0)
                positions_series[pos_name].append(pos_value)

        return {
            'timestamps': np.array(timestamps),
            'angles': {k: np.array(v) for k, v in angles_series.items()},
            'positions': {k: np.array(v) for k, v in positions_series.items()},
            'sequence_length': len(pose_sequence)
        }

    def _generate_feedback(self, rep_analyses: List[Dict], total_errors: List[str]) -> str:
        """Generate user feedback based on analysis"""
        if not rep_analyses:
            return "No repetitions detected. Make sure you perform complete squats with full range of motion."

        feedback_parts = []

        # Rep count feedback
        rep_count = len(rep_analyses)
        feedback_parts.append(f"Detected {rep_count} squat repetition{'s' if rep_count != 1 else ''}.")

        # Quality feedback
        avg_score = np.mean([r['score'] for r in rep_analyses])
        if avg_score >= 90:
            feedback_parts.append("Excellent form!")
        elif avg_score >= 70:
            feedback_parts.append("Good form overall.")
        else:
            feedback_parts.append("Form needs improvement.")

        # Error-specific feedback
        error_counts = {}
        for error in total_errors:
            error_counts[error] = error_counts.get(error, 0) + 1

        feedback_messages = self._get_feedback_messages()
        for error, count in error_counts.items():
            if error in feedback_messages:
                feedback_parts.append(f"{feedback_messages[error]} ({count} rep{'s' if count != 1 else ''})")

        return " ".join(feedback_parts)

    def _get_rep_criteria(self) -> Dict:
        """Define squat rep criteria"""
        return {
            'min_depth_angle': self.min_squat_depth,
            'max_knee_asymmetry': self.max_knee_valgus,
            'min_back_angle': 160,
            'min_rep_duration_frames': 15,
            'max_rep_duration_frames': 60
        }

    def _get_error_definitions(self) -> Dict:
        """Define squat form errors"""
        return {
            'squat_not_deep_enough': {
                'description': 'Squat depth insufficient',
                'severity': 'high',
                'detection_method': 'knee_angle_threshold'
            },
            'knee_asymmetry': {
                'description': 'Uneven knee angles',
                'severity': 'medium',
                'detection_method': 'angle_difference'
            },
            'knee_valgus': {
                'description': 'Knees caving inward',
                'severity': 'high',
                'detection_method': 'knee_tracking_ratio'
            },
            'forward_lean': {
                'description': 'Excessive forward lean',
                'severity': 'medium',
                'detection_method': 'back_angle_threshold'
            }
        }

    def _get_feedback_messages(self) -> Dict:
        """Define user feedback messages"""
        return {
            'squat_not_deep_enough': "Try to squat deeper - aim for thighs parallel to ground.",
            'knee_asymmetry': "Keep both knees aligned - avoid favoring one side.",
            'knee_valgus': "Keep knees tracking over toes - don't let them cave inward.",
            'forward_lean': "Keep your chest up and back straight.",
            'ml_detected_knee_cave': "AI detected knee positioning issues.",
            'ml_detected_forward_lean': "AI detected posture problems."
        }
