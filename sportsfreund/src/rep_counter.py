import cv2
from collections import deque
from pose_extractor import PoseExtractor
from model_trainer import ModelTrainer

class RepCounter:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.pose_extractor = PoseExtractor()
        self.trainer = ModelTrainer(exercise_name)
        self.rep_count = 0
        self.phase_history = deque(maxlen=10)
        self.current_phase = "start"
        self.model_loaded = self.trainer.load_models()
        self.phase_sequence = self._get_phase_sequence()
        self.last_rep_phase = None

    def _get_phase_sequence(self):
        sequences = {
            'squat': ['start', 'down', 'bottom', 'up', 'start'],
            'pushup': ['start', 'down', 'bottom', 'up', 'start'],
        }
        return sequences.get(self.exercise_name, ['start', 'down', 'up', 'start'])

    def count_live(self, source=0):
        if not self.model_loaded:
            print(f"No model found for {self.exercise_name}")
            return

        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose_data = self.pose_extractor.extract_from_frame(frame)

            if pose_data is not None:
                phase_detected = self._detect_phase(pose_data)
                rep_completed = self._check_rep_completion(phase_detected)

                frame = self.pose_extractor.visualize_pose(frame, pose_data)
                self._draw_info(frame, phase_detected, rep_completed)

            cv2.imshow(f"{self.exercise_name} Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rep_count = 0
                self.phase_history.clear()
                self.current_phase = "start"

        cap.release()
        cv2.destroyAllWindows()

    def count_video(self, video_path, show_debug=True):
        if not self.model_loaded:
            print(f"No model found for {self.exercise_name}")
            return 0

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pose_data = self.pose_extractor.extract_from_frame(frame)

            if pose_data is not None:
                phase_detected = self._detect_phase(pose_data)
                rep_completed = self._check_rep_completion(phase_detected)

                if show_debug:
                    frame = self.pose_extractor.visualize_pose(frame, pose_data)
                    self._draw_info(frame, phase_detected, rep_completed)
                    cv2.imshow(f"{self.exercise_name} Analysis", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()
        return self.rep_count

    def _detect_phase(self, pose_data):
        if 'phase' not in self.trainer.models:
            return "unknown"

        features = self._extract_features(pose_data)
        phase = self.trainer.predict_phase(features)
        self.phase_history.append(phase)
        return phase

    def _extract_features(self, pose_data):
        features = []
        features.extend(pose_data['raw_landmarks'])
        features.extend(list(pose_data['angles'].values()))
        features.extend(list(pose_data['positions'].values()))
        return features

    def _check_rep_completion(self, current_phase):
        if len(self.phase_history) < 3:
            return False

        if (self.current_phase == "up" and current_phase == "start" and
            self.last_rep_phase != current_phase):
            self.rep_count += 1
            self.last_rep_phase = current_phase
            self.current_phase = current_phase
            return True

        self.current_phase = current_phase
        return False

    def _draw_info(self, frame, phase, rep_completed):
        cv2.putText(frame, f"Exercise: {self.exercise_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {self.rep_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Phase: {phase}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if rep_completed:
            cv2.putText(frame, "REP COMPLETED!", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
