# pose/pose_detector.py
import mediapipe as mp
import cv2
import numpy as np

class PoseDetector:
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def find_pose(self, img, draw=True):
        """
        Finds pose landmarks in an image
        
        Args:
            img (numpy.ndarray): Input image
            draw (bool): Whether to draw landmarks on the image
            
        Returns:
            tuple: (image with landmarks drawn, landmarks)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            mp.solutions.drawing_utils.draw_landmarks(
                img, 
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
        return img, self.results.pose_landmarks
        
    def get_position(self, img, draw=True):
        """
        Gets positions of all landmarks
        
        Args:
            img (numpy.ndarray): Input image
            draw (bool): Whether to draw points on the image
            
        Returns:
            list: List of landmark positions
        """
        landmarks = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy, lm.visibility])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return landmarks
        
    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        
        Args:
            a (list): First point [x, y]
            b (list): Mid point [x, y]
            c (list): End point [x, y]
            
        Returns:
            float: Angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)