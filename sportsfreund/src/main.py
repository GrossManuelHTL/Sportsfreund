"""
Main Entry Point for Exercise Analysis System
Tests the modular workflow with video analysis
"""
import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.exercise_manager import ExerciseManager
from core.feedback_system import FeedbackType


def print_feedback(message: str):
    """Callback for text feedback"""
    print(f"💬 FEEDBACK: {message}")


def print_custom_feedback(feedback_item):
    """Custom feedback handler with emoji indicators"""
    emoji_map = {
        FeedbackType.INFO: "ℹ️",
        FeedbackType.WARNING: "⚠️",
        FeedbackType.ERROR: "❌",
        FeedbackType.SAFETY: "🚨"
    }
    emoji = emoji_map.get(feedback_item.type, "📝")
    print(f"{emoji} {feedback_item.type.value.upper()}: {feedback_item.message}")


def display_pose_info(pose_data):
    """Display current pose angles and positions"""
    if not pose_data:
        return

    angles = pose_data.get('angles', {})
    positions = pose_data.get('positions', {})

    print(f"🦵 Knee Angles: L={angles.get('left_knee', 0):.1f}° R={angles.get('right_knee', 0):.1f}°")
    print(f"🏋️ Hip Height: {positions.get('hip_center', {}).get('y', 0):.3f}")


def draw_enhanced_overlay(frame, pose_data, status, errors):
    """Draw enhanced overlay with detailed information"""
    height, width = frame.shape[:2]

    # Create overlay with semi-transparent background
    overlay = frame.copy()

    # Status panel background (erweitert für mehr Info)
    cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "EXERCISE ANALYSIS", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Current state with color coding
    state_color = (0, 255, 0) if status['state'] != 'standing' else (128, 128, 128)
    cv2.putText(frame, f"State: {status['state']}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    # State stability info
    frames_in_state = status.get('frames_in_state', 0)
    state_confidence = status.get('state_confidence', 0)
    cv2.putText(frame, f"Stability: {frames_in_state}f | Conf: {state_confidence}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Rep counter with highlighting
    rep_color = (0, 255, 0) if status['reps'] > 0 else (255, 255, 255)
    cv2.putText(frame, f"Reps: {status['reps']}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rep_color, 2)

    # Exercise name
    cv2.putText(frame, f"Exercise: {status['exercise']}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Frame counter
    cv2.putText(frame, f"Frame: {status['frame']}", (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Pose data if available
    if pose_data:
        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})

        # Knee angles with average
        left_knee = angles.get('left_knee', 0)
        right_knee = angles.get('right_knee', 0)
        avg_knee = (left_knee + right_knee) / 2

        cv2.putText(frame, f"L Knee: {left_knee:.1f}°", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"R Knee: {right_knee:.1f}°", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Avg: {avg_knee:.1f}°", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Recent state history
        state_history = status.get('state_history', [])
        if state_history:
            history_text = " -> ".join([h['to'] for h in state_history[-3:]])
            cv2.putText(frame, f"History: {history_text}", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Hip position indicator (rechts)
        hip_y = positions.get('hip_center', {}).get('y', 0.5)
        indicator_y = int(height * 0.8)
        indicator_x = int(width * 0.9)

        # Draw hip height indicator
        cv2.line(frame, (indicator_x - 20, indicator_y - 50),
                (indicator_x - 20, indicator_y + 50), (255, 255, 255), 2)

        current_hip_y = int(indicator_y - 50 + (hip_y * 100))
        cv2.circle(frame, (indicator_x - 20, current_hip_y), 5, (0, 255, 0), -1)
        cv2.putText(frame, "Hip", (indicator_x - 35, indicator_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Knee angle indicator (links von Hip)
        knee_indicator_x = int(width * 0.85)
        knee_bar_height = 100
        knee_bar_top = indicator_y - 50

        # Draw knee angle bar (0-180°)
        cv2.line(frame, (knee_indicator_x, knee_bar_top),
                (knee_indicator_x, knee_bar_top + knee_bar_height), (255, 255, 255), 2)

        # Mark current angle
        normalized_angle = min(180, max(0, avg_knee)) / 180
        current_angle_y = int(knee_bar_top + knee_bar_height - (normalized_angle * knee_bar_height))
        cv2.circle(frame, (knee_indicator_x, current_angle_y), 4, (255, 255, 0), -1)

        # Angle thresholds markers
        cv2.line(frame, (knee_indicator_x - 5, knee_bar_top + int(knee_bar_height * 0.39)),
                (knee_indicator_x + 5, knee_bar_top + int(knee_bar_height * 0.39)), (0, 0, 255), 2)  # 110°
        cv2.line(frame, (knee_indicator_x - 5, knee_bar_top + int(knee_bar_height * 0.22)),
                (knee_indicator_x + 5, knee_bar_top + int(knee_bar_height * 0.22)), (255, 165, 0), 2)  # 140°
        cv2.line(frame, (knee_indicator_x - 5, knee_bar_top + int(knee_bar_height * 0.11)),
                (knee_indicator_x + 5, knee_bar_top + int(knee_bar_height * 0.11)), (0, 255, 0), 2)  # 160°

        cv2.putText(frame, "Knee°", (knee_indicator_x - 20, indicator_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Error display
    if errors:
        error_y = 300
        cv2.rectangle(frame, (10, error_y), (width - 10, error_y + 30 * len(errors) + 10),
                     (0, 0, 128), -1)

        for i, error in enumerate(errors):
            color = (0, 0, 255) if error['type'] == 'error' else (0, 255, 255)
            cv2.putText(frame, f"⚠ {error['message']}", (20, error_y + 25 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    instructions = [
        "Controls:",
        "Q - Quit",
        "R - Reset",
        "SPACE - Pause"
    ]

    start_y = height - 100
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (width - 150, start_y + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def analyze_video(video_path: str, exercise_name: str):
    """Analyze a video file for exercise recognition with enhanced visualization"""

    # Initialize exercise manager
    manager = ExerciseManager(config_dir="exercises")

    # Set up feedback callbacks
    manager.set_feedback_callbacks(
        text_callback=print_feedback,
        custom_callback=print_custom_feedback
    )

    # Load exercise
    if not manager.set_exercise(exercise_name):
        print(f"Failed to load exercise: {exercise_name}")
        return

    # Open video
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Analyzing video: {video_path}")
    print(f"🏃 Exercise: {exercise_name}")
    print(f"📊 Video info: {total_frames} frames @ {fps:.1f} FPS")
    print("=" * 50)

    frame_count = 0
    last_rep_count = 0
    paused = False

    # Create window
    cv2.namedWindow('Exercise Analysis')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("🏁 End of video reached")
                break

            frame_count += 1

        if frame_count % 2 == 0 or paused:

            if not paused:
                result = manager.process_frame(frame)

                if "error" in result:
                    print(f"⚠️ Frame {frame_count}: {result['error']}")
                    cv2.putText(frame, f"Error: {result['error']}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    status = manager.get_current_status()

                    if status["reps"] > last_rep_count:
                        print(f"🎉 REP COMPLETED! Total reps: {status['reps']}")
                        last_rep_count = status["reps"]

                    pose_data = result.get("pose_data")
                    if pose_data:
                        frame = manager.pose_extractor.draw_pose(frame, pose_data)

                    frame = draw_enhanced_overlay(frame, pose_data, status, [])

                    if frame_count % 60 == 0:
                        print(f"\n📊 Frame {frame_count}/{total_frames} | State: {status['state']} | Reps: {status['reps']}")
                        display_pose_info(pose_data)

            # Add pause indicator
            if paused:
                cv2.putText(frame, "PAUSED - Press SPACE to continue",
                           (frame.shape[1]//2 - 150, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Progress bar
            if total_frames > 0:
                progress = frame_count / total_frames
                bar_width = frame.shape[1] - 20
                bar_height = 10
                bar_y = frame.shape[0] - 30

                cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + bar_height),
                             (100, 100, 100), -1)
                cv2.rectangle(frame, (10, bar_y), (10 + int(bar_width * progress), bar_y + bar_height),
                             (0, 255, 0), -1)

                # Progress text
                cv2.putText(frame, f"{progress:.1%}", (10, bar_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Show frame
            cv2.imshow('Exercise Analysis', frame)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("🛑 User requested quit")
            break
        elif key == ord('r'):
            manager.reset_current_exercise()
            last_rep_count = 0
            print("🔄 Exercise state reset")
        elif key == ord(' '):
            paused = not paused
            print(f"⏸️ {'Paused' if paused else 'Resumed'}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final results
    final_status = manager.get_current_status()
    print("\n" + "=" * 50)
    print("🏁 ANALYSIS COMPLETE")
    print(f"📈 Total Reps Detected: {final_status['reps']}")
    print(f"🎯 Final State: {final_status['state']}")
    print(f"📦 Total Frames Processed: {frame_count}")
    print(f"⏱️ Processing Rate: {frame_count/total_frames*100:.1f}% of frames")

    # Zeige finales Feedback basierend auf gesammelten Fehlern
    print("\n" + "=" * 50)
    print("📋 FORM FEEDBACK SUMMARY")
    if manager.error_checker:
        final_feedback = manager.error_checker.get_final_feedback()
        for feedback_msg in final_feedback:
            print(f"  {feedback_msg}")

    print("=" * 50)

    # Deliver any pending feedback
    manager.feedback_handler.deliver_pending_feedback()


def interactive_mode():
    """Interactive mode to select exercise and video"""
    manager = ExerciseManager(config_dir="exercises")

    print("🏋️ EXERCISE ANALYSIS SYSTEM")
    print("=" * 40)

    # Show available exercises
    exercises = manager.get_exercise_list()
    if not exercises:
        print("❌ No exercises found in config directory")
        return

    print("📋 Available exercises:")
    for i, exercise in enumerate(exercises, 1):
        print(f"  {i}. {exercise}")

    # Select exercise
    try:
        choice = int(input(f"\n🎯 Select exercise (1-{len(exercises)}): ")) - 1
        if choice < 0 or choice >= len(exercises):
            print("❌ Invalid choice")
            return
        selected_exercise = exercises[choice]
    except ValueError:
        print("❌ Invalid input")
        return

    # Show available test videos
    video_dir = Path("testvideos")
    if not video_dir.exists():
        print("❌ testvideos directory not found")
        return

    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print("❌ No MP4 videos found in testvideos directory")
        return

    print(f"\n🎬 Available test videos:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")

    # Select video
    try:
        choice = int(input(f"\n🎥 Select video (1-{len(videos)}): ")) - 1
        if choice < 0 or choice >= len(videos):
            print("❌ Invalid choice")
            return
        selected_video = str(videos[choice])
    except ValueError:
        print("❌ Invalid input")
        return

    # Start analysis
    analyze_video(selected_video, selected_exercise)


def main():
    """Main function - entry point"""
    print("🚀 Starting Exercise Analysis System")

    # Check if video and exercise provided as arguments
    if len(sys.argv) >= 3:
        video_path = sys.argv[1]
        exercise_name = sys.argv[2]
        analyze_video(video_path, exercise_name)
    else:
        # Run in interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
