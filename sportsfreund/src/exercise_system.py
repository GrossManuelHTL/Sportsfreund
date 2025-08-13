"""
Main Exercise Analysis System
Central orchestration of video analysis, training, and feedback
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from core.video_analyzer import VideoAnalyzer
from training.exercise_trainer import ExerciseTrainer
from exercises.squat_analyzer import SquatAnalyzer
from ai.ml_models import MLRepDetectionModel, MLFormValidationModel

class ExerciseAnalysisSystem:
    """Main system for exercise analysis and training"""

    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        self.available_exercises = {
            'squat': SquatAnalyzer
        }

    def analyze_video(self, video_path: str, exercise_type: str,
                     use_models: bool = True, show_visualization: bool = False) -> Dict[str, Any]:
        """
        Analyze a video for the specified exercise type

        Args:
            video_path: Path to video file
            exercise_type: Type of exercise ('squat', 'pushup', etc.)
            use_models: Whether to use trained ML models
            show_visualization: Whether to show real-time analysis
        """
        if exercise_type not in self.available_exercises:
            raise ValueError(f"Exercise type '{exercise_type}' not supported. Available: {list(self.available_exercises.keys())}")

        print(f"Analyzing {exercise_type} video: {video_path}")

        # Analyze video and extract pose data
        if show_visualization:
            video_data = self.video_analyzer.analyze_video_with_visualization(video_path)
        else:
            video_data = self.video_analyzer.analyze_video(
                video_path,
                progress_callback=self._print_progress
            )

        if not video_data['pose_sequence']:
            return {
                'success': False,
                'error': 'No pose data could be extracted from video',
                'rep_count': 0,
                'feedback': 'Could not detect person in video'
            }

        # Initialize exercise analyzer
        analyzer = self.available_exercises[exercise_type]()

        # Load trained models if available and requested
        if use_models:
            self._load_models_for_analyzer(analyzer, exercise_type)

        # Perform exercise analysis
        analysis_results = analyzer.analyze_rep(video_data['pose_sequence'])

        # Combine results
        return {
            'success': True,
            'exercise_type': exercise_type,
            'video_metadata': video_data['metadata'],
            'rep_count': analysis_results['rep_count'],
            'rep_analyses': analysis_results.get('rep_analyses', []),
            'errors_detected': analysis_results.get('total_errors', []),
            'feedback': analysis_results['feedback'],
            'average_quality': analysis_results.get('average_quality', 0),
            'exercise_config': analyzer.get_exercise_config()
        }

    def train_exercise_models(self, config_path: str) -> Dict[str, Any]:
        """
        Train ML models for an exercise based on configuration

        Args:
            config_path: Path to training configuration JSON file
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        exercise_name = config['exercise_name']
        print(f"Starting training for {exercise_name}")

        # Initialize trainer
        trainer = ExerciseTrainer(exercise_name)

        # Train models
        training_results = trainer.train_from_config(config_path)

        print(f"Training completed for {exercise_name}")
        return training_results

    def get_available_exercises(self) -> List[str]:
        """Get list of available exercise types"""
        return list(self.available_exercises.keys())

    def register_exercise(self, exercise_name: str, analyzer_class):
        """Register a new exercise analyzer"""
        self.available_exercises[exercise_name] = analyzer_class
        print(f"Registered new exercise: {exercise_name}")

    def _load_models_for_analyzer(self, analyzer, exercise_type: str):
        """Load trained models for the analyzer"""
        model_dir = Path(f"models/{exercise_type}")

        # Load rep detection model
        rep_model_path = model_dir / "rep_model.pkl"
        if rep_model_path.exists():
            analyzer.rep_model.load_model(str(rep_model_path))
            print(f"Loaded rep detection model for {exercise_type}")

        # Load form validation model
        form_model_path = model_dir / "form_model.pkl"
        if form_model_path.exists():
            analyzer.form_model.load_model(str(form_model_path))
            print(f"Loaded form validation model for {exercise_type}")

    def _print_progress(self, progress: float, current: int, total: int):
        """Print analysis progress"""
        percent = int(progress * 100)
        print(f"Analysis progress: {percent}% ({current}/{total} frames)")

class FeedbackGenerator:
    """Generates user-friendly feedback from analysis results"""

    @staticmethod
    def generate_detailed_feedback(analysis_result: Dict) -> str:
        """Generate comprehensive feedback text"""
        if not analysis_result['success']:
            return f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"

        feedback_parts = []

        # Header
        exercise_type = analysis_result['exercise_type'].title()
        rep_count = analysis_result['rep_count']
        feedback_parts.append(f"=== {exercise_type} Analysis Results ===")
        feedback_parts.append(f"Repetitions detected: {rep_count}")

        if rep_count == 0:
            feedback_parts.append("No complete repetitions were detected.")
            feedback_parts.append("Make sure to perform full range of motion exercises.")
            return "\n".join(feedback_parts)

        # Quality overview
        avg_quality = analysis_result.get('average_quality', 0)
        quality_text = FeedbackGenerator._get_quality_description(avg_quality)
        feedback_parts.append(f"Overall quality: {quality_text} ({avg_quality:.1f}/1.0)")

        # Error summary
        errors = analysis_result.get('errors_detected', [])
        if errors:
            error_counts = {}
            for error in errors:
                error_counts[error] = error_counts.get(error, 0) + 1

            feedback_parts.append("\nForm issues detected:")
            for error, count in error_counts.items():
                error_desc = FeedbackGenerator._get_error_description(error)
                feedback_parts.append(f"  • {error_desc} ({count} rep{'s' if count != 1 else ''})")
        else:
            feedback_parts.append("\nExcellent form! No major issues detected.")

        # Individual rep feedback
        rep_analyses = analysis_result.get('rep_analyses', [])
        if len(rep_analyses) > 1:
            feedback_parts.append(f"\nIndividual rep scores:")
            for i, rep in enumerate(rep_analyses, 1):
                score = rep.get('score', 0)
                feedback_parts.append(f"  Rep {i}: {score}/100")

        # Main feedback message
        feedback_parts.append(f"\n{analysis_result['feedback']}")

        return "\n".join(feedback_parts)

    @staticmethod
    def _get_quality_description(quality_score: float) -> str:
        """Convert quality score to descriptive text"""
        if quality_score >= 0.9:
            return "Excellent"
        elif quality_score >= 0.7:
            return "Good"
        elif quality_score >= 0.5:
            return "Fair"
        else:
            return "Needs improvement"

    @staticmethod
    def _get_error_description(error_code: str) -> str:
        """Convert error code to user-friendly description"""
        error_descriptions = {
            'squat_not_deep_enough': 'Squat depth insufficient',
            'knee_asymmetry': 'Uneven knee positioning',
            'knee_valgus': 'Knees caving inward',
            'forward_lean': 'Excessive forward lean',
            'ml_detected_knee_cave': 'AI detected knee alignment issues',
            'ml_detected_forward_lean': 'AI detected posture problems'
        }
        return error_descriptions.get(error_code, error_code.replace('_', ' ').title())
