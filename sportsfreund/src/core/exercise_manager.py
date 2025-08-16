"""
Exercise Manager - Main coordination class
Loads exercise configurations and coordinates all components
"""
import json
import os
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

from .state_machine import StateMachine
from .feedback_system import FeedbackHandler, ErrorChecker, FeedbackType
from .pose_extractor import PoseExtractor


class ExerciseManager:
    """Main class that coordinates exercise recognition, counting and feedback"""

    def __init__(self, config_dir: str = "exercises"):
        self.config_dir = Path(config_dir)
        self.pose_extractor = PoseExtractor()
        self.feedback_handler = FeedbackHandler()

        self.current_exercise = None
        self.state_machine = None
        self.error_checker = None
        self.exercise_configs = {}

        self._load_exercise_configs()

    def _load_exercise_configs(self):
        """Load all exercise configurations from JSON files"""
        if not self.config_dir.exists():
            print(f"Config directory {self.config_dir} does not exist")
            return

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    exercise_name = config.get('name', config_file.stem)
                    self.exercise_configs[exercise_name] = config
                    print(f"Loaded exercise config: {exercise_name}")
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")

    def set_exercise(self, exercise_name: str) -> bool:
        """Set the current exercise"""
        if exercise_name not in self.exercise_configs:
            print(f"Exercise '{exercise_name}' not found")
            return False

        self.current_exercise = exercise_name
        config = self.exercise_configs[exercise_name]

        # Initialize state machine
        self.state_machine = StateMachine(config)

        # Initialize error checker
        self.error_checker = ErrorChecker(config)
        self.error_checker.set_feedback_handler(self.feedback_handler)

        print(f"Exercise set to: {exercise_name}")
        return True

    def process_frame(self, frame) -> Dict[str, Any]:
        """Process a single video frame"""
        if not self.current_exercise or not self.state_machine:
            return {"error": "No exercise selected"}

        # Extract pose data
        pose_data = self.pose_extractor.extract_pose_data(frame)

        if not pose_data:
            return {"error": "No pose detected"}

        # Update state machine
        state_status = self.state_machine.update(pose_data)

        # Check for errors
        errors = self.error_checker.check_errors(pose_data)

        # Add errors to feedback system
        for error in errors:
            self.feedback_handler.add_feedback(
                error.message,
                error.type,
                immediate=(error.type.value == "safety")
            )

        return {
            "pose_data": pose_data,
            "state_status": state_status,
            "errors": [{"message": e.message, "type": e.type.value} for e in errors],
            "exercise": self.current_exercise
        }

    def get_exercise_list(self) -> List[str]:
        """Get list of available exercises"""
        return list(self.exercise_configs.keys())

    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.state_machine:
            return {"exercise": None, "state": None, "reps": 0}

        status = self.state_machine._get_status()
        return {
            "exercise": self.current_exercise,
            "state": status["current_state"],
            "reps": status["rep_count"],
            "frame": status["frame_time"]
        }

    def reset_current_exercise(self):
        """Reset the current exercise state"""
        if self.state_machine:
            self.state_machine.reset()
        self.feedback_handler.pending_feedback.clear()

    def set_feedback_callbacks(self,
                             text_callback: Optional[Callable] = None,
                             audio_callback: Optional[Callable] = None,
                             custom_callback: Optional[Callable] = None):
        """Set callbacks for feedback delivery"""
        if text_callback:
            self.feedback_handler.set_text_callback(text_callback)
        if audio_callback:
            self.feedback_handler.set_audio_callback(audio_callback)
        if custom_callback:
            self.feedback_handler.set_custom_callback(custom_callback)
