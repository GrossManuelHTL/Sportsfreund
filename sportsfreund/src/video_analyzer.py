"""
Video Analysis Pipeline
Processes videos frame by frame and extracts pose data sequences
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from pose_extractor import PoseExtractor

class VideoAnalyzer:
    def __init__(self, frame_skip: int = 2):
        self.pose_extractor = PoseExtractor()
        self.frame_skip = frame_skip

    def analyze_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Analyze a complete video and return pose sequence data

        Returns:
            Dict containing pose sequences, metadata, and analysis results
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        pose_sequence = []
        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for performance
            if frame_count % self.frame_skip == 0:
                pose_data = self.pose_extractor.extract_from_frame(frame)

                if pose_data:
                    pose_data['frame_number'] = frame_count
                    pose_data['timestamp'] = frame_count / fps
                    pose_sequence.append(pose_data)
                    processed_frames += 1

                # Progress callback
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames)

            frame_count += 1

        cap.release()

        return {
            'video_path': video_path,
            'pose_sequence': pose_sequence,
            'metadata': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'fps': fps,
                'duration': total_frames / fps,
                'frame_skip': self.frame_skip
            }
        }

    def analyze_video_with_visualization(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Analyze video with real-time visualization
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

        pose_sequence = []
        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_skip == 0:
                pose_data = self.pose_extractor.extract_from_frame(frame)

                if pose_data:
                    # Visualize pose
                    vis_frame = self.pose_extractor.visualize_pose(frame.copy(), pose_data)

                    # Add frame info
                    cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    pose_data['frame_number'] = frame_count
                    pose_data['timestamp'] = frame_count / fps
                    pose_sequence.append(pose_data)
                    processed_frames += 1

                    # Show frame
                    cv2.imshow('Video Analysis', vis_frame)

                    if writer:
                        writer.write(vis_frame)

                    # Control playback
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)  # Pause

            frame_count += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        return {
            'video_path': video_path,
            'pose_sequence': pose_sequence,
            'metadata': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'fps': fps,
                'duration': total_frames / fps if fps > 0 else 0,
                'frame_skip': self.frame_skip
            }
        }

    def extract_movement_features(self, pose_sequence: List[Dict]) -> Dict:
        """
        Extract movement patterns and features from pose sequence
        """
        if not pose_sequence:
            return {}

        # Extract time series data
        angles_series = {
            'left_knee': [],
            'right_knee': [],
            'left_hip': [],
            'right_hip': [],
            'back': []
        }

        positions_series = {
            'hip_height': [],
            'knee_height': [],
            'ankle_height': []
        }

        timestamps = []

        for frame_data in pose_sequence:
            timestamps.append(frame_data.get('timestamp', 0))

            for angle_name in angles_series:
                angle_value = frame_data['angles'].get(angle_name, 0)
                angles_series[angle_name].append(angle_value)

            for pos_name in positions_series:
                pos_value = frame_data['positions'].get(pos_name, 0)
                positions_series[pos_name].append(pos_value)

        return {
            'timestamps': np.array(timestamps),
            'angles': {k: np.array(v) for k, v in angles_series.items()},
            'positions': {k: np.array(v) for k, v in positions_series.items()},
            'sequence_length': len(pose_sequence)
        }
