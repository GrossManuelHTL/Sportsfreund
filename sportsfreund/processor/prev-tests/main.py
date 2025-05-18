# main.py
import cv2
import argparse
import time
from config import SQUAT_SOURCE
from pose.pose_detector import PoseDetector
from raspyfit.processor.src import process_video
from feedback.exercise_analyzer import ExerciseAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Exercise Analysis using MediaPipe')
    parser.add_argument('--exercise', type=str, default='squat', 
                        choices=['squat', 'pushup', 'lunge'],
                        help='Type of exercise to analyze')
    parser.add_argument('--source', type=str, default=SQUAT_SOURCE,
                        help='Path to video file')
    parser.add_argument('--save', type=str, default='output.mp4',
                        help='Path to save output video')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no display)')
    parser.add_argument('--skip-frames', type=int, default=2,
                        help='Process every N frames (higher = faster but less smooth)')
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2],
                        help='MediaPipe model complexity (0=fastest, 2=most accurate)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of worker threads for parallel processing')
    parser.add_argument('--display-scale', type=float, default=0.75,
                        help='Scale factor for display window (smaller = faster)')
    args = parser.parse_args()
    
    print(f"\nExercise Analysis - {args.exercise.capitalize()}")
    print(f"Performance settings:")
    print(f" - Model complexity: {args.model_complexity}")
    print(f" - Processing every {args.skip_frames} frame(s)")
    print(f" - Using {args.workers} worker threads")
    print(f" - Display scale: {args.display_scale}\n")
    
    start_time = time.time()
    print("Initializing pose detector...")
    pose_detector = PoseDetector(
        model_complexity=args.model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print(f"Initializing {args.exercise} analyzer...")
    exercise_analyzer = ExerciseAnalyzer(exercise_type=args.exercise)
    print(f"Initialization complete ({time.time() - start_time:.2f}s)")
    
    process_video(
        args.source, 
        args.save, 
        pose_detector, 
        exercise_analyzer, 
        headless=args.headless,
        process_every_n_frames=args.skip_frames,
        num_workers=args.workers,
        display_scale=args.display_scale
    )
    
if __name__ == "__main__":
    main()