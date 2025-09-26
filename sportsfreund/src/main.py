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
from models.feedback import FeedbackType
from pipeline.session_manager import SessionManager
from pipeline.audio_system import AudioSystem
import json
from datetime import datetime
from pathlib import Path


def print_feedback(message: str):
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


def resize_frame_for_display(frame, max_width=800, max_height=600):
    """Resize frame to fit within fixed window size while maintaining aspect ratio"""
    height, width = frame.shape[:2]

    # Calculate scaling factor
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height, 1.0)  # Don't upscale

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return frame


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

    # Calculate font scale based on frame size (relative to 800x600 reference)
    font_scale_base = min(width / 800, height / 600)
    font_scale_large = max(0.5, font_scale_base * 0.8)
    font_scale_medium = max(0.4, font_scale_base * 0.6)
    font_scale_small = max(0.3, font_scale_base * 0.4)

    # Calculate thickness based on scale
    thickness_large = max(1, int(font_scale_base * 2))
    thickness_medium = max(1, int(font_scale_base * 1.5))
    thickness_small = max(1, int(font_scale_base * 1))

    # Create overlay with semi-transparent background
    overlay = frame.copy()

    # Status panel background (extended for more info)
    panel_width = int(450 * font_scale_base)
    panel_height = int(280 * font_scale_base)
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "EXERCISE ANALYSIS", (20, int(35 * font_scale_base)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 255), thickness_large)

    # Current state with color coding
    state_color = (0, 255, 0) if status['state'] != 'standing' else (128, 128, 128)
    cv2.putText(frame, f"State: {status['state']}", (20, int(65 * font_scale_base)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, state_color, thickness_medium)

    # State stability info
    frames_in_state = status.get('frames_in_state', 0)
    state_confidence = status.get('state_confidence', 0)
    cv2.putText(frame, f"Stability: {frames_in_state}f | Conf: {state_confidence}", (20, int(85 * font_scale_base)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 200, 200), thickness_small)

    # Rep counter with highlighting
    rep_color = (0, 255, 0) if status['reps'] > 0 else (255, 255, 255)
    cv2.putText(frame, f"Reps: {status['reps']}", (20, int(110 * font_scale_base)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, rep_color, thickness_medium)

    # Exercise name
    cv2.putText(frame, f"Exercise: {status['exercise']}", (20, int(135 * font_scale_base)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness_small)

    # Frame counter
    cv2.putText(frame, f"Frame: {status['frame']}", (20, int(155 * font_scale_base)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness_small)

    # Pose data if available
    if pose_data:
        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})

        # Knee angles with average
        left_knee = angles.get('left_knee', 0)
        right_knee = angles.get('right_knee', 0)
        avg_knee = (left_knee + right_knee) / 2

        cv2.putText(frame, f"L Knee: {left_knee:.1f}°", (20, int(180 * font_scale_base)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 0), thickness_small)
        cv2.putText(frame, f"R Knee: {right_knee:.1f}°", (20, int(200 * font_scale_base)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 0), thickness_small)
        cv2.putText(frame, f"Avg: {avg_knee:.1f}°", (20, int(220 * font_scale_base)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 255, 255), thickness_small)

        # Recent state history
        state_history = status.get('state_history', [])
        if state_history:
            history_text = " -> ".join([h['to'] for h in state_history[-3:]])
            cv2.putText(frame, f"History: {history_text}", (20, int(260 * font_scale_base)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 200, 200), thickness_small)

        # Hip position indicator (right side)
        hip_y = positions.get('hip_center', {}).get('y', 0.5)
        indicator_y = int(height * 0.8)
        indicator_x = int(width * 0.9)

        # Scale indicator elements
        indicator_size = int(50 * font_scale_base)
        line_thickness = max(1, int(2 * font_scale_base))
        circle_radius = max(3, int(5 * font_scale_base))

        # Draw hip height indicator
        cv2.line(frame, (indicator_x - 20, indicator_y - indicator_size),
                (indicator_x - 20, indicator_y + indicator_size), (255, 255, 255), line_thickness)

        current_hip_y = int(indicator_y - indicator_size + (hip_y * indicator_size * 2))
        cv2.circle(frame, (indicator_x - 20, current_hip_y), circle_radius, (0, 255, 0), -1)
        cv2.putText(frame, "Hip", (indicator_x - int(35 * font_scale_base), indicator_y + int(70 * font_scale_base)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness_small)

        # Knee angle indicator (left of Hip)
        knee_indicator_x = int(width * 0.85)
        knee_bar_height = int(100 * font_scale_base)
        knee_bar_top = indicator_y - indicator_size

        # Draw knee angle bar (0-180°)
        cv2.line(frame, (knee_indicator_x, knee_bar_top),
                (knee_indicator_x, knee_bar_top + knee_bar_height), (255, 255, 255), line_thickness)

        # Mark current angle
        normalized_angle = min(180, max(0, avg_knee)) / 180
        current_angle_y = int(knee_bar_top + knee_bar_height - (normalized_angle * knee_bar_height))
        cv2.circle(frame, (knee_indicator_x, current_angle_y), max(2, int(4 * font_scale_base)), (255, 255, 0), -1)

        # Angle thresholds markers
        threshold_size = max(3, int(5 * font_scale_base))
        cv2.line(frame, (knee_indicator_x - threshold_size, knee_bar_top + int(knee_bar_height * 0.39)),
                (knee_indicator_x + threshold_size, knee_bar_top + int(knee_bar_height * 0.39)), (0, 0, 255), line_thickness)  # 110°
        cv2.line(frame, (knee_indicator_x - threshold_size, knee_bar_top + int(knee_bar_height * 0.22)),
                (knee_indicator_x + threshold_size, knee_bar_top + int(knee_bar_height * 0.22)), (255, 165, 0), line_thickness)  # 140°
        cv2.line(frame, (knee_indicator_x - threshold_size, knee_bar_top + int(knee_bar_height * 0.11)),
                (knee_indicator_x + threshold_size, knee_bar_top + int(knee_bar_height * 0.11)), (0, 255, 0), line_thickness)  # 160°

        cv2.putText(frame, "Knee°", (knee_indicator_x - int(20 * font_scale_base), indicator_y + int(70 * font_scale_base)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness_small)

    # Error display
    if errors:
        error_y = int(300 * font_scale_base)
        error_height = int(30 * font_scale_base)
        cv2.rectangle(frame, (10, error_y), (width - 10, error_y + error_height * len(errors) + 10),
                     (0, 0, 128), -1)

        for i, error in enumerate(errors):
            color = (0, 0, 255) if error['type'] == 'error' else (0, 255, 255)
            cv2.putText(frame, f"⚠ {error['message']}", (20, error_y + int(25 * font_scale_base) + i * error_height),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, color, thickness_small)

    # Controls instructions
    instructions = [
        "Controls:",
        "Q - Quit",
        "R - Reset",
        "SPACE - Pause"
    ]

    start_y = height - int(100 * font_scale_base)
    start_x = width - int(150 * font_scale_base)
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (start_x, start_y + i * int(20 * font_scale_base)),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 200, 200), thickness_small)

    return frame


def analyze_video(video_path: str, exercise_name: str, target_reps_per_set: int = 10, target_sets: int = 1):
    """Analyze a video file for exercise recognition with enhanced visualization"""

    # Initialize exercise manager
    manager = ExerciseManager(config_dir="exercises")

    # Initialize audio and session manager for interactive feedback
    audio_system = AudioSystem()
    session_manager = SessionManager(audio_system=audio_system)

    # adapter for custom callback: FeedbackItem -> SessionManager.add_feedback
    def session_feedback_adapter(feedback_item):
        # feedback_item has .type (FeedbackType), .message
        try:
            severity = feedback_item.type.value
            session_manager.add_feedback(feedback_item.type.value, feedback_item.message, severity=severity)
        except Exception:
            pass

    # Start session and the first set with given targets
    session_id = session_manager.start_session(exercise_name, target_sets, target_reps_per_set)
    session_manager.start_set(1)

    # Set up feedback callbacks (text and audio)
    manager.set_feedback_callbacks(
        text_callback=print_feedback,
        audio_callback=audio_system.speak,
        custom_callback=session_feedback_adapter
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
    reps_in_set = 0
    current_set_number = 1

    # Create window
    cv2.namedWindow('Exercise Analysis')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("🏁 End of video reached")
                break

            frame_count += 1

        # Process every 2nd frame for performance, but display all frames
        process_frame = frame_count % 2 == 0 or paused

        if process_frame and not paused:
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
                    # update session manager rep counters
                    reps_in_set += 1
                    session_manager.update_rep_count(reps_in_set)

                    # If set target reached, finish set and aggregate feedback
                    if reps_in_set >= target_reps_per_set:
                        # Finish current set and speak aggregated feedback
                        set_summary = session_manager.finish_set()
                        try:
                            set_feedback = manager.error_checker.on_set_end()
                        except Exception:
                            set_feedback = None

                        # Speak aggregated feedback_texts from error checker (if any)
                        if set_feedback and isinstance(set_feedback, dict):
                            for text in set_feedback.get('feedback_texts', []):
                                audio_system.speak(text, async_play=True)

                        # Deliver any pending feedback (TTS etc.)
                        manager.feedback_handler.deliver_pending_feedback()
                        print(f"Set finished summary: {set_summary}")

                        # Prepare next set if there are more
                        current_set_number += 1
                        reps_in_set = 0
                        if current_set_number <= target_sets:
                            session_manager.start_set(current_set_number)
                        else:
                            # finish session early if reached planned sets
                            break

                pose_data = result.get("pose_data")
                errors = result.get("errors", [])
                if pose_data:
                    frame = manager.pose_extractor.draw_pose(frame, pose_data)

                frame = draw_enhanced_overlay(frame, pose_data, status, errors)

                if frame_count % 60 == 0:
                    print(f"\n📊 Frame {frame_count}/{total_frames} | State: {status['state']} | Reps: {status['reps']}")
                    display_pose_info(pose_data)
        elif not paused:
            # For frames we don't process, still get the current status for overlay
            status = manager.get_current_status()
            frame = draw_enhanced_overlay(frame, None, status, [])

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

        # Resize frame for display
        frame = resize_frame_for_display(frame)

        # Show frame
        cv2.imshow('Exercise Analysis', frame)

        # Handle keyboard input with reduced delay for smoother playback
        key = cv2.waitKey(1) & 0xFF
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

    # Show final feedback based on collected errors
    print("\n" + "=" * 50)
    print("📋 FORM FEEDBACK SUMMARY")
    if manager.error_checker:
        final_feedback = manager.error_checker.get_final_feedback()
        for feedback_msg in final_feedback:
            print(f"  {feedback_msg}")

    print("=" * 50)

    # Finish session and save locally (no backend upload)
    try:
        # prepare payload before finishing session
        payload = session_manager.get_session_data_for_backend()
        session_obj = session_manager.finish_session()

        # save payload locally
        try:
            sessions_dir = Path("sessions")
            sessions_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = payload.get('session', {}).get('session_id', 'unknown')[:8]
            filename = f"session_{sid}_{timestamp}.json"
            with open(sessions_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Session saved locally: {filename}")
        except Exception as e:
            print(f"Could not save session locally: {e}")
    except Exception:
        pass

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
        # optional args: reps_per_set, sets
        reps = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
        sets = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
        analyze_video(video_path, exercise_name, target_reps_per_set=reps, target_sets=sets)
    else:
        # Run in interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
