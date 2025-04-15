import cv2
import time
import numpy as np

def process_video(input_path, output_path, pose_detector, exercise_analyzer):
    """
    Process video for exercise analysis
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save output video
        pose_detector (PoseDetector): Pose detector object
        exercise_analyzer (ExerciseAnalyzer): Exercise analyzer object
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables for FPS calculation
    prev_time = 0
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        current_time = time.time()
        fps_current = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Find pose landmarks
        frame, landmarks = pose_detector.find_pose(frame, draw=False)
        
        if landmarks:
            # Get landmark positions
            landmark_list = pose_detector.get_position(frame, draw=False)
            
            # Analyze exercise
            frame, feedback = exercise_analyzer.analyze(frame, landmark_list, pose_detector)
            
            # Display feedback
            cv2.putText(frame, f"FPS: {int(fps_current)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display rep counter
            cv2.putText(frame, f"Reps: {exercise_analyzer.count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display feedback
            y_pos = 110
            for fb in feedback:
                cv2.putText(frame, fb, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 40
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow("Exercise Analysis", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()