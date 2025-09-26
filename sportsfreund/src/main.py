"""
Main Entry Point for Exercise Analysis System
Minimal version: Analyzes exercise video and shows real-time state and rep count.
"""
import cv2
import sys
import os
from core.exercise_manager import ExerciseManager


def draw_minimal_overlay(frame, state, reps):
    """Draw a minimal overlay with current state and rep count"""
    height, width = frame.shape[:2]
    font_scale = min(width / 800, height / 600) * 0.8
    thickness = max(1, int(font_scale * 2))

    # State
    cv2.putText(frame, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    # Reps
    cv2.putText(frame, f"Reps: {reps}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    return frame


def main():
    """Main function - entry point"""
    print("Starting Exercise Analysis System")

    # Check if video and exercise provided as arguments
    if len(sys.argv) < 3:
        print("Usage: python main.py <video_path> <exercise_name> [target_reps_per_set] [target_sets]")
        sys.exit(1)

    video_path = sys.argv[1]
    exercise_name = sys.argv[2]
    target_reps_per_set = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    target_sets = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)

    # Initialize exercise manager
    manager = ExerciseManager(config_dir="exercises")
    if not manager.set_exercise(exercise_name):
        print(f"Failed to load exercise: {exercise_name}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    cv2.namedWindow('Exercise Analysis')
    frame_count = 0
    last_rep_count = 0
    paused = False
    reps_in_set = 0
    current_set_number = 1

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        process_frame = frame_count % 2 == 0 or paused
        if process_frame and not paused:
            result = manager.process_frame(frame)
            if "error" in result:
                print(f"Error: {result['error']}")
                cv2.putText(frame, f"Error: {result['error']}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                status = manager.get_current_status()
                if status["reps"] > last_rep_count:
                    print(f"REP COMPLETED! Total reps: {status['reps']}")
                    last_rep_count = status["reps"]
                    reps_in_set += 1
                    if reps_in_set >= target_reps_per_set:
                        current_set_number += 1
                        reps_in_set = 0
                        if current_set_number > target_sets:
                            break
                state = status.get('state', '')
                reps = status.get('reps', 0)
                errors = result.get('errors', [])
                for error in errors:
                    print(f"Feedback: {error['message']} ({error['type']})")
                frame = draw_minimal_overlay(frame, state, reps)
        elif not paused:
            status = manager.get_current_status()
            state = status.get('state', '')
            reps = status.get('reps', 0)
            frame = draw_minimal_overlay(frame, state, reps)
        cv2.imshow('Exercise Analysis', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User requested quit")
            break
        elif key == ord('r'):
            manager.reset_current_exercise()
            last_rep_count = 0
            print("Exercise state reset")
        elif key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
    cap.release()
    cv2.destroyAllWindows()
    final_status = manager.get_current_status()
    print(f"Total Reps: {final_status.get('reps', 0)}")
    print(f"Final State: {final_status.get('state', '')}")


if __name__ == "__main__":
    main()
