import cv2
import numpy as np
from collections import deque
from pose_extractor import PoseExtractor
from model_trainer import ModelTrainer

class RepCounter:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.pose_extractor = PoseExtractor()
        self.trainer = ModelTrainer(exercise_name)
        self.rep_count = 0
        self.phase_history = deque(maxlen=5)
        self.model_loaded = self.trainer.load_models()

    def count_live(self, source=0):
        if not self.model_loaded:
            print(f"No model found for {self.exercise_name}")
            return

        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.pose_extractor.extract(frame)

            if landmarks is not None:
                rep_detected = self._detect_rep(landmarks)

                frame = self.pose_extractor.draw(frame)
                self._draw_info(frame, rep_detected)

            cv2.imshow(f"{self.exercise_name} Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rep_count = 0
                self.phase_history.clear()

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
            landmarks = self.pose_extractor.extract(frame)

            if landmarks is not None:
                rep_detected = self._detect_rep(landmarks)

                if show_debug:
                    frame = self.pose_extractor.draw(frame)
                    self._draw_debug_info(frame, frame_count, rep_detected)

                    cv2.imshow(f"{self.exercise_name} Video Analysis", frame)

                    # Pause bei Rep-Detection für bessere Sichtbarkeit
                    wait_time = 500 if rep_detected else 30
                    key = cv2.waitKey(wait_time) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)  # Pause bis nächste Taste

        cap.release()
        if show_debug:
            cv2.destroyAllWindows()
        return self.rep_count

    def _detect_rep(self, landmarks):
        if 'rep' not in self.trainer.models:
            print("WARNING: No rep model found!")
            return False

        X = self.trainer.scalers['rep'].transform(landmarks.reshape(1, -1))
        prediction = self.trainer.models['rep'].predict(X)[0]
        confidence = self.trainer.models['rep'].predict_proba(X)[0].max()

        print(f"Rep prediction: {prediction} (confidence: {confidence:.2f})")

        # Viel strengere Kriterien für Rep-Detection
        if prediction == "rep" and confidence > 0.7:  # Höhere Confidence nötig
            # Zusätzliche Plausibilitätsprüfung über Zeit
            if len(self.phase_history) >= 2:
                # Verhindere zu schnelle Reps (mindestens 2 "no_rep" dazwischen)
                recent_non_reps = sum(1 for x in list(self.phase_history)[-2:] if x == "no_rep")
                if recent_non_reps >= 1:  # Mindestens 1 no_rep Frame dazwischen
                    self.rep_count += 1
                    self.phase_history.append("rep_detected")
                    print(f"🎉 REP {self.rep_count} DETECTED! (confidence: {confidence:.2f})")
                    return True
                else:
                    print(f"Rep ignored - too fast (recent: {list(self.phase_history)[-2:]})")
                    self.phase_history.append("rep_ignored")
            else:
                # Erste Rep im Video
                self.rep_count += 1
                self.phase_history.append("rep_detected")
                print(f"🎉 REP {self.rep_count} DETECTED! (confidence: {confidence:.2f})")
                return True
        else:
            self.phase_history.append("no_rep")
            if prediction == "rep":
                print(f"Rep prediction too weak (confidence: {confidence:.2f} < 0.7)")

        return False

    def _draw_info(self, frame, rep_detected):
        cv2.putText(frame, f"REPS: {self.rep_count}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        if rep_detected:
            cv2.putText(frame, "REP!", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def _draw_debug_info(self, frame, frame_count, rep_detected):
        # Frame-Info
        cv2.putText(frame, f"REPS: {self.rep_count}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Frame: {frame_count}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Model-Prediction anzeigen (letztes Ergebnis)
        if 'rep' in self.trainer.models:
            # Dummy-Prediction für Anzeige (echte wird in _detect_rep gemacht)
            cv2.putText(frame, f"Exercise: {self.exercise_name}", (50, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Rep-Detection Feedback
        if rep_detected:
            cv2.putText(frame, "REP DETECTED!", (50, 180),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
            cv2.circle(frame, (150, 50), 40, (0, 255, 255), 4)

        # Historie anzeigen
        history_text = " -> ".join(list(self.phase_history)[-3:])
        cv2.putText(frame, f"History: {history_text}", (50, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Steuerung
        cv2.putText(frame, "Controls: q=quit, space=pause", (50, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
