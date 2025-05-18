from feedback.feedback_base import ExerciseBase
from utils.visualization import draw_angle, put_text
from config import EXERCISE_PARAMS, COLORS

class SquatAnalyzer(ExerciseBase):
    """
    Squat exercise analyzer
    """
    def __init__(self):
        super().__init__()
        # Load squat-specific parameters from config
        params = EXERCISE_PARAMS["squat"]
        self.min_knee_angle = params["min_knee_angle"]
        self.max_knee_angle = params["max_knee_angle"]
        self.min_hip_angle = params["min_hip_angle"]
        self.max_hip_angle = params["max_hip_angle"]
        self.rep_threshold = params["rep_threshold"]
        
    def analyze(self, image, landmarks, pose_detector):
        """
        Analyze squat form and count repetitions
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            tuple: (image with analysis, feedback list)
        """
        if not landmarks or len(landmarks) < 33:
            return image, ["No pose detected"]
            
        self.form_feedback = []
        
        self.count_repetition(landmarks, pose_detector)
        
        self.check_form(landmarks, pose_detector)
        
        try:
            # Hip, knee and ankle landmarks (right side)
            hip = [landmarks[24][1], landmarks[24][2]]
            knee = [landmarks[26][1], landmarks[26][2]]
            ankle = [landmarks[28][1], landmarks[28][2]]
            shoulder = [landmarks[12][1], landmarks[12][2]]
            
            # Calculate angles
            knee_angle = pose_detector.calculate_angle(hip, knee, ankle)
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
            
            stage_text = f"Stage: {self.stage.upper() if self.stage else 'None'}"
            put_text(image, stage_text, (10, 150), color=COLORS["text"])
            
        except Exception as e:
            self.form_feedback.append(f"Error in squat analysis: {str(e)}")
            
        return image, self.form_feedback
        
    def check_form(self, landmarks, pose_detector):
        """
        Check squat form and provide feedback
        
        Args:
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            list: List of feedback strings
        """
        try:
            hip = [landmarks[24][1], landmarks[24][2]]
            knee = [landmarks[26][1], landmarks[26][2]]
            ankle = [landmarks[28][1], landmarks[28][2]]
            shoulder = [landmarks[12][1], landmarks[12][2]]
            
            knee_angle = pose_detector.calculate_angle(hip, knee, ankle)
            hip_angle = pose_detector.calculate_angle(shoulder, hip, knee)
            
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
                if knee[0] > ankle[0] + 30:
                    self.form_feedback.append("Knees going too far forward")
                    

                back_angle = pose_detector.calculate_angle(
                    [landmarks[11][1], landmarks[11][2]],  # left shoulder
                    [landmarks[23][1], landmarks[23][2]],  # left hip
                    [landmarks[25][1], landmarks[25][2]]   # left knee
                )
                if back_angle < 160:
                    self.form_feedback.append("Keep your back straighter")
                    
            return self.form_feedback
            
        except Exception as e:
            self.form_feedback.append(f"Error analyzing form: {str(e)}")
            return self.form_feedback
            
    def count_repetition(self, landmarks, pose_detector):
        """
        Count squat repetitions
        
        Args:
            landmarks (list): List of landmarks
            pose_detector (PoseDetector): Pose detector object
            
        Returns:
            bool: True if a repetition was counted
        """
        try:
            hip = [landmarks[24][1], landmarks[24][2]]
            knee = [landmarks[26][1], landmarks[26][2]]
            ankle = [landmarks[28][1], landmarks[28][2]]
            
            knee_angle = pose_detector.calculate_angle(hip, knee, ankle)

            print(knee_angle)
            
            if knee_angle > self.max_knee_angle - 10:
                self.stage = "up"
                return False
                
            if knee_angle < self.min_knee_angle + 20 and self.stage == "up":
                self.stage = "down"
                self.count += 1
                return True
                
            return False
            
        except Exception as e:
            self.form_feedback.append(f"Error counting repetition: {str(e)}")
            return False
