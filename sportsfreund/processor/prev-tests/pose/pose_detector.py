# pose/pose_detector.py
import mediapipe as mp
import cv2
import numpy as np
import time

class PoseDetector:
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,  # Can be 0, 1, or 2 (lower is faster)
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 enable_segmentation=False):  # Disable for speed
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Cache for angle calculations
        self.angle_cache = {}
        
    def find_pose(self, img, draw=True):
        """
        Finds pose landmarks in an image
        
        Args:
            img (numpy.ndarray): Input image
            draw (bool): Whether to draw landmarks on the image
            
        Returns:
            tuple: (image with landmarks drawn, landmarks)
        """
        # Check if image is empty
        if img is None or img.size == 0:
            return img, None
            
        # Resize image for faster processing if it's large
        h, w = img.shape[:2]
        original_size = (w, h)
        max_size = 640  # Max width or height for processing
        
        # Only resize if image is larger than max_size
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
                
            process_img = cv2.resize(img, (new_w, new_h))
            was_resized = True
        else:
            process_img = img
            was_resized = False
        
        # Convert to RGB (avoid conversion if possible)
        process_img_rgb = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.pose.process(process_img_rgb)
        
        # Check if pose was detected
        if self.results.pose_landmarks:
            # If we resized for processing, need to scale landmarks back
            if was_resized:
                for landmark in self.results.pose_landmarks.landmark:
                    # Scale coordinates back to original size
                    landmark.x *= (original_size[0] / new_w)
                    landmark.y *= (original_size[1] / new_h)
            
            # Draw landmarks if requested
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, 
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
            return img, self.results.pose_landmarks
        else:
            return img, None
        
    def get_position(self, img, draw=True, cached=True):
        """
        Gets positions of all landmarks
        
        Args:
            img (numpy.ndarray): Input image
            draw (bool): Whether to draw points on the image
            cached (bool): Whether to use cached results (if available)
            
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
                    # Only draw visible landmarks (improves visual clarity)
                    if lm.visibility > 0.5:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return landmarks
        
    def calculate_angle(self, a, b, c, use_cache=True):
        """
        Calculate angle between three points
        
        Args:
            a (list): First point [x, y]
            b (list): Mid point [x, y]
            c (list): End point [x, y]
            use_cache (bool): Whether to use cached angles
            
        Returns:
            float: Angle in degrees
        """
        # Use caching for repeated angle calculations
        if use_cache:
            # Create a hash for the points
            point_hash = hash(f"{a[0]},{a[1]}_{b[0]},{b[1]}_{c[0]},{c[1]}")
            
            # Check if we've already calculated this angle
            if point_hash in self.angle_cache:
                return self.angle_cache[point_hash]
        
        # Calculate vectors
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        # Cache result
        if use_cache:
            self.angle_cache[point_hash] = angle
            
            # Limit cache size to prevent memory issues
            if len(self.angle_cache) > 1000:
                # Remove oldest entries (simple approach)
                for _ in range(100):
                    self.angle_cache.pop(next(iter(self.angle_cache)))
        
        return angle