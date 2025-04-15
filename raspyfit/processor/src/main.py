import cv2
import argparse
from config import SQUAT_SOURCE
from pose.pose_detector import PoseDetector
from utils.visualization import draw_landmarks, put_text
from video.video_processor import process_video
from feedback.exercise_analyzer import ExerciseAnalyzer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Exercise Analysis using MediaPipe')
    parser.add_argument('--exercise', type=str, default='squat', 
                        choices=['squat', 'pushup', 'lunge'],
                        help='Type of exercise to analyze')
    parser.add_argument('--source', type=str, default=SQUAT_SOURCE,
                        help='Path to video file')
    parser.add_argument('--save', type=str, default='output.mp4',
                        help='Path to save output video')
    args = parser.parse_args()
    
    # Initialize pose detector
    pose_detector = PoseDetector()
    
    # Initialize exercise analyzer
    exercise_analyzer = ExerciseAnalyzer(exercise_type=args.exercise)
    
    # Process video
    process_video(args.source, args.save, pose_detector, exercise_analyzer)
    
if __name__ == "__main__":
    main()