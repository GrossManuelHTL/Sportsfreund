import os
import argparse
import cv2
import tempfile
import numpy as np
from model_trainer import ExerciseModelTrainer
from exercise_analyzer import ExerciseAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Fitness-Übungs-Analyse mit KI')
    parser.add_argument('--mode', type=str, default='analyze',
                        choices=['train', 'analyze', 'live', 'both'],
                        help='Betriebsmodus: train, analyze, live oder both')
    parser.add_argument('--training_dir', type=str, default='trainingdata',
                        help='Verzeichnis mit Trainingsvideos')
    parser.add_argument('--test_video', type=str, default=None,
                        help='Zu analysierendes Video')
    parser.add_argument('--model_path', type=str, default='squat_model.h5',
                        help='Pfad zum Modell')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualisierung anzeigen')

    args = parser.parse_args()

    error_categories = [
        "correct",
        "knees_too_far_apart",
        "back_not_straight",
        "too_high",
        "wrong_going_up"
    ]

    if args.mode in ['train', 'both']:
        print(f"=== Starte Training mit Videos aus {args.training_dir} ===")
        trainer = ExerciseModelTrainer(
            error_categories=error_categories,
            model_path=args.model_path
        )
        success = trainer.train_model(args.training_dir)

        if not success:
            print("Training fehlgeschlagen!")
            if args.mode == 'both':
                print("Analyse wird übersprungen.")
                return

    if args.mode == 'live':
        if not os.path.exists(args.model_path):
            print(f"Fehler: Modell {args.model_path} nicht gefunden!")
            return
        analyze_live_webcam(args.model_path, error_categories, args.visualize)
        return

    if args.mode in ['analyze', 'both']:
        if args.test_video is None:
            print("Fehler: Kein Testvideo angegeben!")
            print("Verwende --test_video, um ein Video zu analysieren.")
            return

        if not os.path.exists(args.model_path):
            print(f"Fehler: Modell {args.model_path} nicht gefunden!")
            return

        print(f"=== Analysiere Video {args.test_video} ===")
        analyzer = ExerciseAnalyzer(
            model_path=args.model_path,
            error_categories=error_categories
        )

        feedback = analyzer.analyze_video(args.test_video, show_visualization=args.visualize)

        if feedback:
            print("\n=== Feedback zur Übungsausführung ===")
            print(f"Erkannte Kategorie: {feedback['predicted_category']} (Konfidenz: {feedback['confidence']:.2f})")
            print(f"Feedback: {feedback['text']}")

            print("\nDetaillierte Wahrscheinlichkeiten:")
            for category, prob in feedback['all_probabilities'].items():
                print(f"  - {category}: {prob:.4f}")
        else:
            print("Analyse fehlgeschlagen!")


def analyze_live_webcam(model_path, error_categories, show_visualization=True):
    import mediapipe as mp

    analyzer = ExerciseAnalyzer(
        model_path=model_path,
        error_categories=error_categories
    )

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variables for squat detection
    squat_frames = []
    in_squat = False
    hip_y_baseline = None
    squat_threshold = 0.05
    current_feedback = None
    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame

    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value

    print("Squat analysis mode - perform squats naturally")
    print("q - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Only run pose detection on every nth frame
        process_this_frame = frame_count % process_every_n_frames == 0
        pose_results = None

        if process_this_frame:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            # Draw pose landmarks if enabled
            if show_visualization and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)

            # Detect squat using hip position
            if pose_results.pose_landmarks:
                hip_landmark = pose_results.pose_landmarks.landmark[LEFT_HIP]
                hip_y = hip_landmark.y

                # Initialize baseline if not set
                if hip_y_baseline is None and hip_y > 0:
                    hip_y_baseline = hip_y

                if hip_y_baseline:
                    # Detect squat start (hip goes down)
                    if not in_squat and hip_y > (hip_y_baseline + squat_threshold):
                        in_squat = True
                        squat_frames = []

                    # Detect squat end (hip comes back up)
                    if in_squat and hip_y < (hip_y_baseline + squat_threshold * 0.5):
                        in_squat = False

                        if len(squat_frames) >= 10:
                            # Create temporary video more efficiently
                            temp_fd, temp_video = tempfile.mkstemp(suffix='.mp4')
                            os.close(temp_fd)

                            height, width = squat_frames[0].shape[:2]
                            out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                                  30, (width, height))

                            for f in squat_frames:
                                out.write(f)
                            out.release()

                            # Analyze the video
                            feedback = analyzer.analyze_video(temp_video, show_visualization=False)

                            if feedback:
                                current_feedback = feedback
                                print("\n=== Squat Feedback ===")
                                print(
                                    f"Detected: {feedback['predicted_category']} (Confidence: {feedback['confidence']:.2f})")
                                print(f"Feedback: {feedback['text']}")

                            try:
                                os.unlink(temp_video)
                            except:
                                pass

        # Add current frame to squat frames if in squat (use smaller frame size)
        if in_squat:
            # Store smaller frames to reduce memory usage
            small_frame = cv2.resize(frame, (320, 240))
            squat_frames.append(small_frame)

        # Display current feedback (simplified to reduce rendering overhead)
        if current_feedback:
            cv2.putText(frame, f"{current_feedback['predicted_category']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show only first line of feedback
            main_feedback = current_feedback['text'].split('\n')[0]
            cv2.putText(frame, main_feedback, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        cv2.imshow('Exercise Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


def analyze_single_video(video_path, model_path, visualize=False):
    error_categories = [
        "correct",
        "knees_too_far_apart",
        "back_not_straight",
        "too_high",
        "wrong_going_up"
    ]

    analyzer = ExerciseAnalyzer(
        model_path=model_path,
        error_categories=error_categories
    )

    return analyzer.analyze_video(video_path, show_visualization=visualize)


if __name__ == "__main__":
    main()