"""
Base classes for exercise analysis
Defines the interface for exercise-specific analyzers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class ExerciseAnalyzer(ABC):
    """Abstract base class for all exercise analyzers"""
    
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.rep_count = 0
        self.detected_errors = []
        
    @abstractmethod
    def analyze_rep(self, pose_sequence: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a pose sequence to detect reps and form errors
        
        Args:
            pose_sequence: List of pose data dictionaries
            
        Returns:
            Analysis results including rep count and error feedback
        """
        pass
    
    @abstractmethod
    def detect_rep_boundaries(self, movement_features: Dict) -> List[tuple]:
        """
        Detect start and end points of individual repetitions
        
        Args:
            movement_features: Extracted movement patterns
            
        Returns:
            List of (start_frame, end_frame) tuples for each rep
        """
        pass
    
    @abstractmethod
    def validate_rep_form(self, rep_data: Dict) -> Dict[str, Any]:
        """
        Validate the form of a single repetition
        
        Args:
            rep_data: Data for one repetition
            
        Returns:
            Form validation results and error feedback
        """
        pass
    
    def get_exercise_config(self) -> Dict:
        """Return exercise-specific configuration"""
        return {
            'name': self.exercise_name,
            'rep_criteria': self._get_rep_criteria(),
            'error_definitions': self._get_error_definitions(),
            'feedback_messages': self._get_feedback_messages()
        }
    
    @abstractmethod
    def _get_rep_criteria(self) -> Dict:
        """Define what constitutes a valid repetition"""
        pass
    
    @abstractmethod
    def _get_error_definitions(self) -> Dict:
        """Define common form errors for this exercise"""
        pass
    
    @abstractmethod
    def _get_feedback_messages(self) -> Dict:
        """Define feedback messages for different scenarios"""
        pass

class RepDetectionModel(ABC):
    """Abstract base class for rep detection models"""
    
    @abstractmethod
    def train(self, training_data: List[Dict], labels: List[int]) -> Dict:
        """Train the model with labeled data"""
        pass
    
    @abstractmethod
    def predict(self, pose_data: Dict) -> Dict[str, float]:
        """Make predictions on new pose data"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load trained model from file"""
        pass

class FormValidationModel(ABC):
    """Abstract base class for form validation models"""
    
    @abstractmethod
    def train(self, training_data: List[Dict], error_labels: List[str]) -> Dict:
        """Train the model with labeled error data"""
        pass
    
    @abstractmethod
    def validate_form(self, pose_data: Dict) -> Dict[str, Any]:
        """Validate form and detect errors"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load trained model from file"""
        pass
