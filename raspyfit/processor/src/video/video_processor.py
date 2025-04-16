import cv2
import time
import numpy as np
import threading
from queue import Queue, Full

class FrameProcessor(threading.Thread):
    """Thread class for parallel frame processing"""
    def __init__(self, input_queue, output_queue, pose_detector, exercise_analyzer):
        threading.Thread.__init__(self)
        self.daemon = True
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.pose_detector = pose_detector
        self.exercise_analyzer = exercise_analyzer
        self.running = True
        
    def run(self):
        while self.running:
            try:
                frame_data = self.input_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                    
                frame, frame_index = frame_data
                
                processed_frame, landmarks = self.pose_detector.find_pose(frame.copy(), draw=False)
                
                if landmarks:
                    landmark_list = self.pose_detector.get_position(processed_frame, draw=False)
                    
                    processed_frame, feedback = self.exercise_analyzer.analyze(
                        processed_frame, landmark_list, self.pose_detector)
                    
                    cv2.putText(processed_frame, f"Reps: {self.exercise_analyzer.count}", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    y_pos = 110
                    for fb in feedback:
                        cv2.putText(processed_frame, fb, (10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        y_pos += 40
                self.output_queue.put((processed_frame, frame_index))
                
                self.input_queue.task_done()
                
            except Exception as e:
                print(f"Error in processing thread: {e}")
                if self.input_queue.qsize() > 0:
                    self.input_queue.task_done()
    
    def stop(self):
        self.running = False

def process_video(input_path, output_path, pose_detector, exercise_analyzer, headless=False, 
                  process_every_n_frames=2, num_workers=2, display_scale=0.75):
    """
    Process video for exercise analysis using parallel processing
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save output video
        pose_detector (PoseDetector): Pose detector object
        exercise_analyzer (ExerciseAnalyzer): Exercise analyzer object
        headless (bool): Whether to run in headless mode (no display)
        process_every_n_frames (int): Process every Nth frame (skip frames for better speed)
        num_workers (int): Number of parallel processing threads
        display_scale (float): Scale factor for display window (smaller is faster)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    display_width = int(frame_width * display_scale)
    display_height = int(frame_height * display_scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / process_every_n_frames, 
                          (frame_width, frame_height))
    
    frame_queue = Queue(maxsize=32)
    result_queue = Queue()
    
    workers = []
    for _ in range(num_workers):
        worker = FrameProcessor(frame_queue, result_queue, pose_detector, exercise_analyzer)
        worker.start()
        workers.append(worker)
    
    prev_time = time.time()
    start_time = prev_time
    frame_count = 0
    processed_count = 0
    
    processed_frames = {}
    next_frame_to_write = 0
    
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Processing every {process_every_n_frames} frame(s) with {num_workers} worker(s)")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % process_every_n_frames != 0:
                continue
                
            current_time = time.time()
            elapsed = current_time - prev_time
            prev_time = current_time
            
            if frame_count % 30 == 0:
                elapsed_total = current_time - start_time
                percent_done = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Reading: {frame_count}/{total_frames} frames ({percent_done:.1f}%), "
                      f"Queue size: {frame_queue.qsize()}, "
                      f"Elapsed: {elapsed_total:.1f}s")
            
            try:
                frame_queue.put((frame, processed_count), block=True, timeout=2.0)
                processed_count += 1
            except Full:
                print("Warning: Processing queue is full, skipping frame")
                
            while not result_queue.empty():
                try:
                    result_frame, frame_idx = result_queue.get_nowait()
                    processed_frames[frame_idx] = result_frame
                except Exception:
                    break
                    
            while next_frame_to_write in processed_frames:
                out_frame = processed_frames[next_frame_to_write]
                
                fps_current = 1.0 / elapsed if elapsed > 0 else 0
                cv2.putText(out_frame, f"FPS: {int(fps_current)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out.write(out_frame)
                
                if not headless:
                    try:
                        display_frame = cv2.resize(out_frame, (display_width, display_height))
                        cv2.imshow("Exercise Analysis", display_frame)
                        
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q'):
                            print("User requested exit")
                            break
                    except Exception as e:
                        print(f"Warning: Could not display frame: {str(e)}")
                        print("Switching to headless mode...")
                        headless = True

                del processed_frames[next_frame_to_write]
                next_frame_to_write += 1
                
        frame_queue.join()
        
        while not result_queue.empty():
            result_frame, frame_idx = result_queue.get()
            processed_frames[frame_idx] = result_frame
            
        for idx in sorted(processed_frames.keys()):
            out.write(processed_frames[idx])
            
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    finally:
        for worker in workers:
            worker.stop()
            frame_queue.put(None) 
            
        cap.release()
        out.release()
        if not headless:
            cv2.destroyAllWindows()
            
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Video processing complete.")
        print(f"Total frames: {frame_count}, Processed frames: {processed_count}")
        print(f"Total time: {total_time:.1f}s, Average FPS: {processed_count/total_time:.1f}")
        print(f"Output saved to: {output_path}")