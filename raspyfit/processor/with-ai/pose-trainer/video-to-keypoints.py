import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm


class PoseExtractor:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_video(self, video_path):
        """Extract pose keypoints from a video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        # Process each frame
        keypoints_data = []
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            # Extract keypoints from the frame
            keypoints = self.process_frame(frame)
            if keypoints is not None:
                keypoints_data.append(keypoints)

        cap.release()

        if not keypoints_data:
            print(f"Warning: No pose keypoints detected in {video_path}")
            return None

        return np.array(keypoints_data)

    def process_frame(self, frame):
        """Process a single frame and extract pose keypoints."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(frame_rgb)

        # Extract landmarks if detected
        if results.pose_landmarks:
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                  for lm in results.pose_landmarks.landmark])
            return landmarks

        return None

    def close(self):
        """Close the pose detector."""
        self.pose.close()


def process_videos(input_dir, output_dir, exercise_name=None):
    """
    Process all videos in the input directory and save keypoints to the output directory.

    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save extracted keypoints
        exercise_name: If provided, only process videos with this name in filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pose extractor
    extractor = PoseExtractor()

    # Get list of video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in video_extensions)]

    if exercise_name:
        video_files = [f for f in video_files if exercise_name.lower() in f.lower()]

    if not video_files:
        print(f"No video files found in {input_dir}")
        if exercise_name:
            print(f"With exercise name: {exercise_name}")
        return

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)

        # Extract exercise name from filename if not provided
        current_exercise = exercise_name
        if not current_exercise:
            # Try to extract exercise name from the filename
            # Assuming format: exercise_name_quality.mp4 (e.g., squat_correct.mp4)
            filename = os.path.splitext(video_file)[0]
            parts = filename.split('_')
            if len(parts) >= 1:
                current_exercise = parts[0]

        # Determine if this is a "correct" example
        is_correct = "correct" in video_file.lower()
        quality = "correct" if is_correct else "incorrect"

        try:
            # Extract keypoints
            keypoints = extractor.extract_from_video(video_path)

            if keypoints is not None:
                # Create filename for the keypoints
                output_filename = f"{current_exercise}_{quality}_{len(keypoints)}.npy"
                output_path = os.path.join(output_dir, output_filename)

                # Save keypoints
                np.save(output_path, keypoints)
                print(f"Saved {len(keypoints)} frames to {output_path}")
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    # Close the extractor
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pose keypoints from videos')
    parser.add_argument('--input', type=str, required=False,
                        help='Input directory containing video files', default='../data/videos')
    parser.add_argument('--output', type=str, default='../data/poses',
                        help='Output directory for extracted keypoints')
    parser.add_argument('--exercise', type=str, default=None,
                        help='Exercise name (optional, extracts from filename if not provided)')

    args = parser.parse_args()

    process_videos(args.input, args.output, args.exercise)