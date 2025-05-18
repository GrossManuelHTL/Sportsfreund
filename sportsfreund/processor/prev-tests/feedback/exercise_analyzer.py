import cv2
import numpy as np
from utils.visualization import draw_angle, put_text

class ExerciseAnalyzer:
    def __init__(self, exercise_type="squat"):
        self.exercise_type = exercise_type
        self.count = 0
        self.direction = 0  # 0: neutral, 1: going down, 2: going up
        self.form_feedback = []
        
        # Exercise-specific parameters
        if exercise_type == "squat":
            self.min_knee_angle = 90
            self.max_knee_angle = 170
            self.min_hip_angle = 80
            self.max_hip_angle = 170
            self.stage = None  # up or down
            
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
        self.form_feedback = []
        
        if self.exercise_type == "squat":
            return self._analyze_squat(image, landmarks, pose_detector)
        elif self.exercise_type == "pushup":
            # Future implementation
            return image, ["Pushup analysis not implemented yet"]
        elif self.exercise_type == "lunge":
            # Future implementation
            return image, ["Lunge analysis not implemented yet"]
        else:
            return image, ["Unknown exercise type"]
            
    def _analyze_squat(self, image, landmarks, pose_detector):
        """
        Analyze squat form and count repetitions
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            tuple: (image with analysis, feedback list)
        """
        if not landmarks or len(landmarks) < 33:  # MediaPipe has 33 landmarks
            return image, ["No pose detected"]
            
        # Extract relevant landmarks for squat analysis
        # Hip, knee and ankle landmarks (right side)
        try:
            hip = [landmarks[24][1], landmarks[24][2]]
            knee = [landmarks[26][1], landmarks[26][2]]
            ankle = [landmarks[28][1], landmarks[28][2]]
            
            # Calculate knee angle (between hip, knee, ankle)
            knee_angle = pose_detector.calculate_angle(hip, knee, ankle)
            
            # Calculate hip angle (between shoulder, hip, knee)
            shoulder = [landmarks[12][1], landmarks[12][2]]
            hip_angle = pose_detector.calculate_angle(shoulder, hip, knee)
            
            # Draw angles on image
            image = draw_angle(image, 
                                (hip[0], hip[1]), 
                                (knee[0], knee[1]), 
                                (ankle[0], ankle[1]), 
                                knee_angle)
            
            image = draw_angle(image, 
                                (shoulder[0], shoulder[1]), 
                                (hip[0], hip[1]), 
                                (knee[0], knee[1]), 
                                hip_angle)
            
            # Count repetitions
            if knee_angle > self.max_knee_angle - 10:
                self.stage = "up"
            if knee_angle < self.min_knee_angle + 20 and self.stage == "up":
                self.stage = "down"
                self.count += 1
                
            # Form feedback
            if self.stage == "down":
                # Check knee angle
                if knee_angle < self.min_knee_angle:
                    self.form_feedback.append("Knees too bent")
                elif knee_angle > self.min_knee_angle + 30:
                    self.form_feedback.append("Squat deeper")
                    
                # Check hip angle
                if hip_angle < self.min_hip_angle:
                    self.form_feedback.append("Hips too low")
                elif hip_angle > self.min_hip_angle + 30:
                    self.form_feedback.append("Bend at the hips more")
                    
                # Check if knees are going beyond toes
                # Simplified check: if knee x-coord is much further than ankle x-coord
                if knee[0] > ankle[0] + 30:
                    self.form_feedback.append("Knees going too far forward")
                    
            # Display squat stage
            put_text(image, f"Stage: {self.stage.upper() if self.stage else 'None'}", 
                     (10, 150), color=(255, 255, 0))
            
            return image, self.form_feedback
            
        except Exception as e:
            return image, [f"Error in squat analysis: {str(e)}"]