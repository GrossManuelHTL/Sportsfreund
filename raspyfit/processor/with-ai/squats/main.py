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
    import numpy as np
    import time

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

    # Define custom landmarks drawing spec
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

    # Define key landmarks for squat analysis
    LANDMARKS = {
        'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP.value,
        'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE.value,
        'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE.value,
        'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE.value,
        'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open webcam")
        return

    # Get webcam dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Status variables for exercise tracking
    current_prediction = None
    prediction_confidence = 0.0
    squat_stage = "stand"  # Can be "stand" or "squat"
    squat_counter = 0

    # Depth tracking variables
    min_knee_angle = 180  # Track lowest angle during squat
    depth_threshold = 110  # Good depth is below this angle

    # Feedback collection (only show after squat completion)
    current_squat_feedback = {}
    show_feedback_until = 0  # Timestamp until when to show feedback
    feedback_display_time = 3  # seconds to display feedback after squat
    feedback_to_show = ""

    print("Starting live webcam analysis. Press 'q' to quit.")

    def calculate_angle(a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def draw_angle(frame, a, b, c, text_pos, color=(255, 255, 0)):
        """Draw the angle between three points on the frame"""
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape
        ax, ay = int(a.x * w), int(a.y * h)
        bx, by = int(b.x * w), int(b.y * h)
        cx, cy = int(c.x * w), int(c.y * h)

        # Draw lines
        cv2.line(frame, (bx, by), (ax, ay), color, 2)
        cv2.line(frame, (bx, by), (cx, cy), color, 2)

        # Calculate angle
        angle = calculate_angle(a, b, c)

        # Draw angle text
        cv2.putText(frame, f"{angle:.1f}°",
                    (int(text_pos.x * w), int(text_pos.y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return angle

    current_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB and process every frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        # Update current time
        current_time = time.time()

        # Process the frame only if landmarks were detected
        if pose_results.pose_landmarks:
            # Draw pose landmarks if enabled
            if show_visualization:
                # Basic landmark drawing
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec,
                    connection_drawing_spec)

                # Draw key angles
                landmarks = pose_results.pose_landmarks.landmark
                h, w, _ = frame.shape

                # Draw vertical reference line
                mid_x = w // 2
                cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 0, 255), 1)

                # Label key points
                for name, idx in LANDMARKS.items():
                    pos = landmarks[idx]
                    px, py = int(pos.x * w), int(pos.y * h)
                    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                    cv2.putText(frame, name.lower().replace('_', ' '),
                                (px + 5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Calculate and draw knee angles
                left_knee_angle = draw_angle(
                    frame,
                    landmarks[LANDMARKS['LEFT_HIP']],
                    landmarks[LANDMARKS['LEFT_KNEE']],
                    landmarks[LANDMARKS['LEFT_ANKLE']],
                    landmarks[LANDMARKS['LEFT_KNEE']]
                )

                right_knee_angle = draw_angle(
                    frame,
                    landmarks[LANDMARKS['RIGHT_HIP']],
                    landmarks[LANDMARKS['RIGHT_KNEE']],
                    landmarks[LANDMARKS['RIGHT_ANKLE']],
                    landmarks[LANDMARKS['RIGHT_KNEE']]
                )

                # Calculate back angle (vertical alignment)
                left_back_angle = draw_angle(
                    frame,
                    landmarks[LANDMARKS['LEFT_SHOULDER']],
                    landmarks[LANDMARKS['LEFT_HIP']],
                    landmarks[LANDMARKS['LEFT_KNEE']],
                    landmarks[LANDMARKS['LEFT_HIP']],
                    color=(255, 0, 0)
                )

                right_back_angle = draw_angle(
                    frame,
                    landmarks[LANDMARKS['RIGHT_SHOULDER']],
                    landmarks[LANDMARKS['RIGHT_HIP']],
                    landmarks[LANDMARKS['RIGHT_KNEE']],
                    landmarks[LANDMARKS['RIGHT_HIP']],
                    color=(255, 0, 0)
                )

                # Add overlay with angle information
                cv2.rectangle(frame, (10, h - 120), (250, h - 10), (0, 0, 0), -1)
                cv2.putText(frame, f"Left Knee: {left_knee_angle:.1f}°",
                            (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Right Knee: {right_knee_angle:.1f}°",
                            (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Back Angle: {left_back_angle:.1f}°",
                            (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw hip-ankle vertical alignment
                left_hip_x = int(landmarks[LANDMARKS['LEFT_HIP']].x * w)
                left_ankle_x = int(landmarks[LANDMARKS['LEFT_ANKLE']].x * w)
                cv2.line(frame,
                         (left_ankle_x, int(landmarks[LANDMARKS['LEFT_ANKLE']].y * h)),
                         (left_ankle_x, int(landmarks[LANDMARKS['LEFT_HIP']].y * h)),
                         (0, 255, 255), 1)

            # Perform squat analysis using the model
            try:
                landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                prediction = analyzer.predict_from_landmarks(landmarks_array)

                if prediction:
                    current_prediction = prediction['predicted_category']
                    prediction_confidence = prediction['confidence']

                    # Calculate average knee angle
                    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                    # Update minimum knee angle during squat
                    if squat_stage == "squat" and avg_knee_angle < min_knee_angle:
                        min_knee_angle = avg_knee_angle

                    # Store feedback during squat (don't show it yet)
                    # Don't collect "too_high" during squat - will evaluate at end
                    if current_prediction != "correct" and current_prediction != "too_high" and prediction_confidence > 0.65:
                        # Record this issue if not already recorded or if confidence is higher
                        if (current_prediction not in current_squat_feedback or
                                prediction_confidence > current_squat_feedback[current_prediction]):
                            current_squat_feedback[current_prediction] = prediction_confidence

                    if show_visualization:
                        # Display squat form status (not feedback)
                        cv2.putText(frame, f"Form: {current_prediction}",
                                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {prediction_confidence:.2f}",
                                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Squat counter logic and depth tracking
                    if avg_knee_angle < 120 and squat_stage == "stand":
                        squat_stage = "squat"
                        min_knee_angle = avg_knee_angle  # Reset min angle for new squat
                    elif avg_knee_angle > 160 and squat_stage == "squat":
                        squat_stage = "stand"
                        squat_counter += 1

                        # Check depth at the end of squat
                        if min_knee_angle > depth_threshold:
                            current_squat_feedback["too_high"] = 1.0  # Add depth issue

                        # Squat completed - prepare feedback for display
                        if current_squat_feedback:
                            # Format the feedback text
                            issues = [analyzer.generate_feedback_text(category)
                                      for category in current_squat_feedback.keys()]
                            feedback_to_show = f"Squat #{squat_counter} - Issues found:\n" + "\n".join(issues)
                        else:
                            feedback_to_show = f"Squat #{squat_counter} completed - Great form!"

                        # Add depth info to feedback
                        feedback_to_show += f"\nLowest knee angle: {min_knee_angle:.1f}°"

                        # Set time to display feedback
                        show_feedback_until = current_time + feedback_display_time

                        # Clear for next squat
                        current_squat_feedback = {}

                    if show_visualization:
                        # Display squat counter and depth info
                        cv2.putText(frame, f"Squats: {squat_counter}",
                                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if squat_stage == "squat":
                            cv2.putText(frame, f"Min angle: {min_knee_angle:.1f}°",
                                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during analysis: {e}")

        # Add feedback text if we should be displaying it and visualization is enabled
        if current_time < show_feedback_until and show_visualization and feedback_to_show:
            # Create a semi-transparent overlay for the feedback text
            overlay = frame.copy()

            # Split feedback into multiple lines
            feedback_lines = feedback_to_show.split('\n')
            feedback_height = len(feedback_lines) * 30  # Adjust height based on number of lines

            cv2.rectangle(overlay, (0, height - 50 - feedback_height),
                          (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Display each line of feedback
            for i, line in enumerate(feedback_lines):
                cv2.putText(frame, line,
                            (10, height - 20 - (len(feedback_lines) - 1 - i) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Exercise Analysis', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam analysis completed")


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