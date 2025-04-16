import cv2
import numpy as np
from abc import ABC, abstractmethod
from utils.visualization import draw_angle, put_text

class ExerciseBase(ABC):
    """
    Base class for all exercise analyzers
    """
    def __init__(self):
        self.count = 0
        self.stage = None
        self.form_feedback = []
        
    @abstractmethod
    def analyze(self, image, landmarks, pose_detector):
        """
        Analyze exercise form and count repetitions
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            tuple: (image with analysis, feedback list)
        """
        pass
        
    @abstractmethod
    def check_form(self, landmarks, pose_detector):
        """
        Check exercise form and provide feedback
        
        Args:
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            list: List of feedback strings
        """
        pass
        
    @abstractmethod
    def count_repetition(self, landmarks, pose_detector):
        """
        Count exercise repetitions
        
        Args:
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            bool: True if a repetition was counted
        """
        pass
        
    def reset(self):
        """Reset exercise counter and stage"""
        self.count = 0
        self.stage = None
        self.form_feedback = []
        
    def get_count(self):
        """Get current repetition count"""
        return self.count
        
    def get_stage(self):
        """Get current exercise stage"""
        return self.stage
        
    def get_feedback(self):
        """Get form feedback"""
        return self.form_feedback