import os
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import time
from pose_extractor import PoseExtractor
from tensorflow.keras.models import load_model


class ExerciseAnalyzer:
    def __init__(self, exercise_name=None, model_dir="models"):
        """
        Initialize the exercise analyzer

        Args:
            exercise_name (str): Name of the exercise to analyze
            model_dir (str): Directory where models are stored
        """
        from exercise_manager import ExerciseManager

        self.model_dir = model_dir
        self.manager = ExerciseManager()
        self.exercise_name = exercise_name
        self.pose_extractor = None
        self.model = None

        if exercise_name:
            config = self.manager.get_exercise_config(exercise_name)
            if config:
                self.pose_extractor = PoseExtractor(config)
                model_path = os.path.join(model_dir, self.pose_extractor.get_model_name())
                if os.path.exists(model_path):
                    self.model = load_model(model_path)
                    print(f"Loaded model for {exercise_name} from {model_path}")
                else:
                    print(f"No trained model found at {model_path}")
            else:
                print(f"No configuration found for exercise: {exercise_name}")

    def predict_from_landmarks(self, landmarks_array):
        """
        Predicts exercise form from landmarks using the trained model

        Args:
            landmarks_array: numpy array of landmark positions

        Returns:
            Dictionary with predicted category and confidence
        """
        try:
            # Get the expected input shape from the model
            expected_shape = self.model.input_shape

            # Model expects (None, 30, 32) but we have a single frame (1, 99)
            # Need to reshape to match expected format
            num_landmarks = 33  # MediaPipe provides 33 landmarks
            num_coords = 3  # x, y, z coordinates per landmark

            # Reshape the landmarks to match the expected format
            # For sequence models (expected_shape has 3 dimensions)
            if len(expected_shape) > 2:
                seq_length = expected_shape[1]  # Usually 30
                features = expected_shape[2]  # Usually 32

                # Create a sequence by duplicating the current frame
                # Not ideal but allows us to use the model for real-time feedback
                flat_landmarks = landmarks_array.flatten()

                # Make sure we don't exceed the feature size
                if len(flat_landmarks) > features:
                    flat_landmarks = flat_landmarks[:features]
                elif len(flat_landmarks) < features:
                    # Pad with zeros if we have fewer features than expected
                    flat_landmarks = np.pad(flat_landmarks, (0, features - len(flat_landmarks)))

                reshaped_data = np.tile(flat_landmarks, (seq_length, 1))
                model_input = np.expand_dims(reshaped_data, axis=0)  # Add batch dimension
            else:
                # For models that expect flattened input
                flat_landmarks = landmarks_array.flatten()
                model_input = np.expand_dims(flat_landmarks, axis=0)

            # Make prediction with properly shaped input
            prediction = self.model.predict(model_input, verbose=0)[0]

            # Get categories from the config
            categories = [category["id"] for category in self.pose_extractor.config["categories"]]

            predicted_idx = np.argmax(prediction)
            confidence = prediction[predicted_idx]

            return {
                "predicted_category": categories[predicted_idx],
                "confidence": float(confidence)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def analyze_video(self, video_path, show_visualization=False):
        """
        Analyze exercise form in a video

        Args:
            video_path (str): Path to the video file
            show_visualization (bool): Whether to show visualization

        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None

        if not self.pose_extractor or not self.model:
            print("Pose extractor or model not initialized")
            return None

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None

        frame_count = 0
        predictions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = pose.process(image)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = np.array([[landmark.x, landmark.y, landmark.z]
                                      for landmark in results.pose_landmarks.landmark])

                # Get prediction for this frame
                prediction = self.predict_from_landmarks(landmarks)
                if prediction:
                    predictions.append(prediction)

                # Visualization if requested
                if show_visualization:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    cv2.putText(frame, f"Prediction: {prediction['predicted_category']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {prediction['confidence']:.2f}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow('Video Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        if show_visualization:
            cv2.destroyAllWindows()

        # Process all predictions to get overall result
        if not predictions:
            print("No pose landmarks detected in the video")
            return None

        # Find most common prediction
        categories = {}
        for pred in predictions:
            category = pred["predicted_category"]
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1

        # Get category with most occurrences
        predicted_category = max(categories, key=categories.get)

        # Calculate confidence (percentage of frames with this prediction)
        confidence = categories[predicted_category] / len(predictions)

        return {
            "exercise": self.exercise_name,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "frame_count": frame_count
        }

    def generate_feedback_text(self, category_id):
        """Generate feedback text for a given category ID"""
        for category in self.pose_extractor.config["categories"]:
            if str(category["id"]) == str(category_id):
                return category["feedback"]
        return "No specific feedback available."

    def analyze_live(self, config_or_exercise_name):
        """
        Analyze exercise form in real-time using webcam input

        Args:
            config_or_exercise_name: Either an exercise name string or a config dictionary
        """
        # Handle whether we got a string exercise name or a config dict
        if isinstance(config_or_exercise_name, str):
            exercise_config = self.manager.get_exercise_config(config_or_exercise_name)
        else:
            exercise_config = config_or_exercise_name  # Use the config directly

        # Update pose extractor with the config
        self.pose_extractor = PoseExtractor(exercise_config)
        self.model_path = os.path.join(self.model_dir, self.pose_extractor.get_model_name())
        self.model = load_model(self.model_path)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Set up webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Get landmarks dictionary from config's relevant_landmarks
        LANDMARKS = {}
        for landmark_name in self.pose_extractor.config["relevant_landmarks"]:
            try:
                LANDMARKS[landmark_name] = getattr(mp_pose.PoseLandmark, landmark_name)
            except AttributeError:
                print(f"Warning: Landmark {landmark_name} not found in MediaPipe")

        # Find the "not deep enough" category from config
        correct_category_id = "correct"
        not_deep_enough_id = next(
            (cat["id"] for cat in self.pose_extractor.config["categories"]
             if "deep" in cat["id"] or "tief" in cat["id"].lower()),
            "not_deep_enough"
        )

        # Tracking variables
        rep_counter = 0
        rep_stage = "stand"  # stand or squat
        min_knee_angle = 180
        current_rep_feedback = {}
        feedback_to_show = ""
        show_feedback_until = 0
        feedback_display_time = 5  # seconds
        depth_threshold = 100  # Adjust based on exercise

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and get pose landmarks
            results = pose.process(rgb_frame)

            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate angles
                def calculate_angle(a, b, c):
                    a = np.array([a.x, a.y])
                    b = np.array([b.x, b.y])
                    c = np.array([c.x, c.y])
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(np.degrees(radians))
                    return 360 - angle if angle > 180 else angle

                # Calculate relevant angles (example for squats)
                left_knee_angle = calculate_angle(
                    landmarks[LANDMARKS['LEFT_HIP']],
                    landmarks[LANDMARKS['LEFT_KNEE']],
                    landmarks[LANDMARKS['LEFT_ANKLE']]
                ) if all(k in LANDMARKS for k in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']) else 180

                right_knee_angle = calculate_angle(
                    landmarks[LANDMARKS['RIGHT_HIP']],
                    landmarks[LANDMARKS['RIGHT_KNEE']],
                    landmarks[LANDMARKS['RIGHT_ANKLE']]
                ) if all(k in LANDMARKS for k in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']) else 180

                left_back_angle = calculate_angle(
                    landmarks[LANDMARKS['LEFT_SHOULDER']],
                    landmarks[LANDMARKS['LEFT_HIP']],
                    landmarks[LANDMARKS['LEFT_KNEE']]
                ) if all(k in LANDMARKS for k in ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE']) else 180

                # Draw pose landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Add angle overlays
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10, h - 120), (250, h - 10), (0, 0, 0), -1)
                cv2.putText(frame, f"Left Knee: {left_knee_angle:.1f}°",
                            (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Right Knee: {right_knee_angle:.1f}°",
                            (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Back Angle: {left_back_angle:.1f}°",
                            (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Perform analysis
                try:
                    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                    prediction = self.predict_from_landmarks(landmarks_array)

                    if prediction:
                        current_prediction = prediction['predicted_category']
                        prediction_confidence = prediction['confidence']

                        # Calculate average knee angle
                        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                        # Update minimum knee angle during rep
                        if rep_stage == "squat" and avg_knee_angle < min_knee_angle:
                            min_knee_angle = avg_knee_angle

                        # Store feedback during rep (don't show it yet)
                        if current_prediction != correct_category_id and current_prediction != not_deep_enough_id and prediction_confidence > 0.65:
                            if (current_prediction not in current_rep_feedback or
                                    prediction_confidence > current_rep_feedback[current_prediction]):
                                current_rep_feedback[current_prediction] = prediction_confidence

                        # Display form status
                        cv2.putText(frame, f"Form: {current_prediction}",
                                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {prediction_confidence:.2f}",
                                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Repetition counter logic (example for squats)
                        if avg_knee_angle < 120 and rep_stage == "stand":
                            rep_stage = "squat"
                            min_knee_angle = avg_knee_angle  # Reset min angle for new rep
                        elif avg_knee_angle > 160 and rep_stage == "squat":
                            rep_stage = "stand"
                            rep_counter += 1

                            # Check depth at the end of rep
                            if min_knee_angle > depth_threshold:
                                current_rep_feedback[not_deep_enough_id] = 1.0  # Add depth issue

                            # Rep completed - prepare feedback
                            if current_rep_feedback:
                                # Format the feedback text
                                issues = [self.generate_feedback_text(category)
                                          for category in current_rep_feedback.keys()]
                                feedback_to_show = f"Rep #{rep_counter} - Issues found:\n" + "\n".join(issues)
                            else:
                                feedback_to_show = f"Rep #{rep_counter} completed - Great form!"

                            # Add depth info
                            feedback_to_show += f"\nLowest knee angle: {min_knee_angle:.1f}°"

                            # Set time to display feedback
                            show_feedback_until = current_time + feedback_display_time

                            # Clear for next rep
                            current_rep_feedback = {}

                        # Display rep counter
                        cv2.putText(frame, f"Reps: {rep_counter}",
                                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if rep_stage == "squat":
                            cv2.putText(frame, f"Min angle: {min_knee_angle:.1f}°",
                                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error during analysis: {e}")

            # Add feedback text if applicable
            if current_time < show_feedback_until and feedback_to_show:
                overlay = frame.copy()

                # Split feedback into multiple lines
                feedback_lines = feedback_to_show.split('\n')
                feedback_height = len(feedback_lines) * 30

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