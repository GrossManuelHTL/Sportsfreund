import cv2
import numpy as np
import mediapipe as mp

def draw_landmarks(image, landmarks, connections=None, color=(0, 255, 0), thickness=2):
    """
    Draw landmarks and connections on image
    
    Args:
        image (numpy.ndarray): Input image
        landmarks (list): List of landmarks
        connections (list): List of connections
        color (tuple): Color in BGR
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with landmarks drawn
    """
    h, w, c = image.shape
    
    if not landmarks:
        return image
        
    # Draw landmarks
    for lm in landmarks:
        cx, cy = int(lm[1]), int(lm[2])
        cv2.circle(image, (cx, cy), 7, color, cv2.FILLED)
        
    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][1]), int(landmarks[start_idx][2]))
                end_point = (int(landmarks[end_idx][1]), int(landmarks[end_idx][2]))
                cv2.line(image, start_point, end_point, color, thickness)
                
    return image

def draw_angle(image, p1, p2, p3, angle, color=(255, 255, 255), thickness=2):
    """
    Draw angle between three points
    
    Args:
        image (numpy.ndarray): Input image
        p1, p2, p3 (tuple): Points coordinates (x, y)
        angle (float): Angle in degrees
        color (tuple): Color in BGR
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with angle drawn
    """
    # Draw lines
    cv2.line(image, p1, p2, color, thickness)
    cv2.line(image, p2, p3, color, thickness)
    
    # Put angle text
    mid_x = p2[0]
    mid_y = p2[1] - 15
    cv2.putText(image, f"{int(angle)}°", (mid_x, mid_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
    
    return image

def put_text(image, text, position, color=(255, 255, 255), font_scale=1, thickness=2):
    """
    Put text on image
    
    Args:
        image (numpy.ndarray): Input image
        text (str): Text to put
        position (tuple): Position (x, y)
        color (tuple): Color in BGR
        font_scale (float): Font scale
        thickness (int): Text thickness
        
    Returns:
        numpy.ndarray: Image with text
    """
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)
    
    return image