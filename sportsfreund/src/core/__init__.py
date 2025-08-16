"""
Core module for exercise recognition system
"""
from .pose_extractor import PoseExtractor
from .state_machine import StateMachine
from .feedback_system import FeedbackHandler, ErrorChecker, FeedbackType
from .exercise_manager import ExerciseManager

__all__ = [
    'PoseExtractor',
    'StateMachine',
    'FeedbackHandler',
    'ErrorChecker',
    'FeedbackType',
    'ExerciseManager'
]
