import time
import os
import cv2
import logging
import argparse
from enum import Enum
from main import ExerciseManager
from instructions.audio_instruction import InstructionExplainer

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WorkoutState(Enum):
    WAITING_FOR_SELECTION = 0
    EXPLAINING = 1
    COUNTDOWN = 2
    EXERCISING = 3
    FEEDBACK = 4
    COMPLETED = 5

class InteractiveTrainer:
    def __init__(self, exercise=None, camera_index=0, debug=True):
        self.camera_index = camera_index
        self.debug = debug
        self.exercise_manager = None
        self.state = WorkoutState.WAITING_FOR_SELECTION
        self.current_exercise = None
        self.countdown_seconds = 5
        self.reps_per_feedback = 4  # Feedback alle 4 Wiederholungen
        self.reps_goal = 10
        self.exercise = exercise
        self.instruction_explainer = None
        self.no_person_counter = 0
        self.no_person_threshold = 10
        self.error_wait = -200
        self.last_played = None
        self.current_feedback = None
        self.feedback_during_rep = []
        self.wrong_reps = 0

    def list_exercises(self):
        """Listet alle verfügbaren Übungen aus dem exercises-Ordner auf"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        exercises_dir = os.path.join(base_dir, 'exercises')

        available_exercises = []
        for item in os.listdir(exercises_dir):
            item_path = os.path.join(exercises_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
                available_exercises.append(item)

        return available_exercises

    def explain_exercise(self, exercise_name):
        """Erklärt die Übung mit Audioanweisungen"""
        self.state = WorkoutState.EXPLAINING
        print(f"\n=== Übungsanleitung: {exercise_name} ===")

        self.instruction_explainer = InstructionExplainer(exercise_name)

        print("\nVorbereitung der Übung:")
        self.instruction_explainer.speak_prep_instruction()

        print("\nAusführung der Übung:")
        self.instruction_explainer.speak_exec_instruction()

        return True

    def _generate_exercise_cues(self, exercise_name, phase):
        """Generiert passende Ansagen für jede Phase der Übung"""
        cues = {
            "squats": {
                "down": ["Jetzt nach unten gehen", "In die Hocke gehen", "Runter"],
                "up": ["Wieder hochkommen", "Aufrichten", "Nach oben druecken", "Hoch"],
                "bottom": ["Halte diese Position", "Halte die tiefe Position"]
            },
            "push_ups": {
                "down": ["Senke dich nach unten", "Brust Richtung Boden", "Runter"],
                "up": ["Druecke dich hoch", "Nach oben druecken", "Hoch"],
                "bottom": ["Halte diese Position", "Halte die untere Position"]
            }
        }

        if exercise_name in cues and phase in cues[exercise_name]:
            # Wähle zufällig eine der Ansagen
            import random
            return random.choice(cues[exercise_name][phase])
        return phase  # Fallback: Phase-Name zurückgeben

    def start_exercise(self, exercise_name, reps_goal=None):
        """Startet die Übung nach Countdown"""
        if reps_goal:
            self.reps_goal = reps_goal

        # Initialisiere ExerciseManager
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'exercises', exercise_name, 'config.json')

        self.exercise_manager = ExerciseManager(
            exercise_name=exercise_name,
            config_path=config_path,
            debug=self.debug
        )

        self.feedback_map = self.exercise_manager.full_feedback_map
        logging.info(self.feedback_map)
        self.feedback_tts_description = {
            key: value['description_long'] for key, value in self.feedback_map.items()
        }

        logging.info(self.feedback_tts_description)
        self.feedback_priority = {
            key: value['priority'] for key, value in self.feedback_map.items()
        }
        logging.info(self.feedback_priority)
        self.feedback_counters = {key: 0 for key in self.feedback_map}
        logging.info(self.feedback_counters)
        self.feedback_thresholds = {
            key: value['threshold'] for key, value in self.feedback_map.items()
        }
        logging.info(self.feedback_thresholds)

        self.current_exercise = exercise_name
        self.current_rep = 0
        self.last_phase = None
        self.last_feedback_rep = 0

        # Webcam starten und Übung durchführen
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            logging.error(f"Fehler beim Öffnen der Webcam mit Index {self.camera_index}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_time = time.time()
        frame_number = 0

        print(f"\nÜbung gestartet: {exercise_name}")
        print(f"Ziel: {self.reps_goal} Wiederholungen")
        print("Drücke 'q' zum Beenden")

        self.exercise_manager.analyzer.reset()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Frame spiegeln für natürlichere Anzeige
            frame = cv2.flip(frame, 1)

            # Berechne FPS
            frame_number += 1
            elapsed_time = time.time() - start_time
            fps = frame_number / elapsed_time if elapsed_time > 0 else 0

            # Verarbeite Frame mit dem Pose-Extraktor
            landmarks, processed_frame = self.exercise_manager.pose_extractor.extract_landmarks(frame)

            if landmarks is None:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Keine Person erkannt!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.no_person_counter += 1
                if self.no_person_counter >= self.no_person_threshold:
                    self.instruction_explainer.say_sentence_no_wait("Keine Person erkannt! Bitte stellen sie sich seitlich vor die Kamera.")
                    self.no_person_counter = self.error_wait
                    self.last_played = "no_person"
                cv2.imshow('Interaktives Training', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            elif self.last_played == "no_person":
                self.instruction_explainer.stop_speaking()
                self.no_person_counter = 0
            else:
                self.no_person_counter = 0

            coords = self.exercise_manager.pose_extractor.get_landmark_coordinates(landmarks, width, height)
            joint_angles = self.exercise_manager.pose_extractor.calculate_joint_angles(coords)

            # Verarbeite Frame mit dem Übungs-Analysator
            analyzed_frame, status, feedback_keys = self.exercise_manager.analyzer.analyze_frame(processed_frame, joint_angles, coords, self.debug)

            for key in self.feedback_counters:
                if key in feedback_keys:
                    self.feedback_counters[key] += 1
                elif self.last_played == key:
                    #self.instruction_explainer.stop_speaking()
                    self.feedback_counters[key] = 0
                    self.last_played = None
                else:
                    self.feedback_counters[key] = 0

            logging.info("Feedback_keys: " + ', '.join(feedback_keys))
            logging.info("Feedback_during_rep " + ', '.join(self.feedback_during_rep))

            for key in feedback_keys:
                if (key not in self.feedback_during_rep
                        and self.feedback_counters.get(key, 50) >= self.feedback_thresholds.get(key, 0)):
                    logging.info("aojdnjalsd")
                    self.feedback_during_rep.append(key)

            for key in self.feedback_counters:
                count = self.feedback_counters.get(key, 0)
                if count >= self.feedback_thresholds.get(key, 50):
                    if self.feedback_priority.get(key, 100) < self.feedback_priority.get(self.current_feedback, 100):
                        self.current_feedback = key
                    self.feedback_counters[key] = self.error_wait

            if self.current_feedback is not None:
                feedback_text = self.feedback_tts_description.get(self.current_feedback, "Unbekanntes Problem")
                #print(f"\nFeedback: {feedback_text}")
                self.instruction_explainer.say_sentence_no_wait(feedback_text)
                self.last_played = self.current_feedback
                #logging.info(self.current_feedback)
                self.current_feedback = None

            # Aktuelle Wiederholungszahl und Ziel anzeigen
            cv2.putText(analyzed_frame, f"Wiederholung: {self.exercise_manager.analyzer.rep_count}/{self.reps_goal}",
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Sprachansagen für Phasenwechsel
            current_phase = self.exercise_manager.analyzer.current_phase
            #logging.info(feedback_keys)

            if current_phase != self.last_phase and current_phase is not None:
                cue = self._generate_exercise_cues(exercise_name, current_phase)
                #print(f"Phase: {current_phase} - {cue}")
                if self.last_phase == self.exercise_manager.analyzer.phases[0] and current_phase == self.exercise_manager.analyzer.phases[1]:
                    self.feedback_during_rep = []
                self.last_phase = current_phase

            # Rep-Counter prüfen
            if self.exercise_manager.analyzer.rep_count > self.current_rep:

                #idx = self.feedback_during_rep.index('good_form') if 'good_form' in self.feedback_during_rep else 0
                #list = self.feedback_during_rep[idx:]   #Rep starts with good_form, because thats the starting pos and everything before that is to find the start pos and is no rep feedback
                if len(self.feedback_during_rep) == 1 and self.feedback_during_rep[0] == 'good_form':
                    self.current_rep = self.exercise_manager.analyzer.rep_count
                    print(f"Wiederholung {self.current_rep} abgeschlossen!")
                    self.instruction_explainer.say_sentence_no_wait(f"Wiederholung {self.current_rep}")

                    # # Bei jeder 4. Wiederholung Feedback geben
                    # if self.current_rep % self.reps_per_feedback == 0 and self.current_rep > self.last_feedback_rep:
                    #     self._give_feedback()
                    #     self.last_feedback_rep = self.current_rep

                    # Ziel erreicht?
                    if self.current_rep >= self.reps_goal:
                        print("\nZiel erreicht! Super gemacht!")
                        self.instruction_explainer.say_sentence_no_wait("Ziel erreicht! Super gemacht!")
                        time.sleep(3)  # Zeit zum Feiern
                        break
                else:
                    self.wrong_reps += 1
                    self.exercise_manager.analyzer.rep_count -= 1

                logging.info(self.feedback_during_rep)
                #logging.info(self.wrong_reps)
                self.feedback_during_rep = []

            cv2.imshow('Interaktives Training', analyzed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\nÜbung beendet.")
        print(f"Ausgeführte Wiederholungen: {self.current_rep}")

        self.state = WorkoutState.COMPLETED
        return self.exercise_manager.analyzer.rep_count

    def _give_feedback(self):
        """Gibt Feedback zur Übungsausführung basierend auf der Feedback-Historie"""
        self.state = WorkoutState.FEEDBACK

        # Analysiere die bisherige Feedback-Historie
        feedback_history = self.exercise_manager.analyzer.feedback_history

        if not feedback_history:
            return

        # Zähle die Häufigkeit jeder Feedback-Art
        from collections import Counter
        feedback_counter = Counter(feedback_history[-10:])  # Betrachte die letzten 10 Feedbacks
        most_common_feedback = feedback_counter.most_common(2)  # Die beiden häufigsten Probleme

        # Toleranteres Feedback - nur wenn ein Problem wirklich häufig auftritt
        total_feedbacks = len(feedback_history[-10:])
        for feedback_id, count in most_common_feedback:
            # Nur Feedback geben wenn das Problem in mehr als 40% der Fälle auftritt
            if count / total_feedbacks > 0.4 and feedback_id != 'good_form':
                feedback_text = self.exercise_manager.analyzer.feedback_map[feedback_id][0]
                print(f"\nFeedback: {feedback_text}")
                self.instruction_explainer.say_sentence_no_wait(feedback_text)
                return

        # Wenn die Ausführung gut ist
        if 'good_form' in feedback_counter:
            good_feedback = "Gut gemacht! Weiter so!"
            print(f"\nFeedback: {good_feedback}")
            self.instruction_explainer.say_sentence_no_wait(good_feedback)

def main():
    """
    Hauptfunktion für die Kommandozeilenausführung.
    """
    parser = argparse.ArgumentParser(description='Interaktiver Fitness-Trainer')

    parser.add_argument('--exercise', type=str, default=None,
                        help='Name der Übung')
    parser.add_argument('--camera', type=int, default=0,
                        help='Index der Webcam (0 für Standardkamera)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug-Modus aktivieren')
    parser.add_argument('--reps', type=int, default=-1,
                        help='Anzahl der Wiederholungen')

    args = parser.parse_args()

    trainer = InteractiveTrainer(
        exercise=args.exercise,
        camera_index=args.camera,
        debug=args.debug
    )

    try:
        # Wähle direkt eine Übung und starte
        if args.exercise:
            available_exercises = trainer.list_exercises()
            if args.exercise in available_exercises:
                print(f"Starte direkt mit Übung: {args.exercise}")

                if args.reps > 0:
                    reps_goal = args.reps
                else:
                    # Anzahl der Wiederholungen
                    reps_goal = int(input(f"Wie viele Wiederholungen möchtest du machen? (Standard: {trainer.reps_goal}): ") or trainer.reps_goal)

                trainer.explain_exercise(args.exercise)
                trainer.start_exercise(args.exercise, reps_goal)
            else:
                print(f"Übung '{args.exercise}' nicht gefunden!")
        else:
            print("\n=== Willkommen beim interaktiven Fitnesstrainer! ===\n")
            available_exercises = trainer.list_exercises()

            print("Verfügbare Übungen:")
            for i, ex in enumerate(available_exercises, 1):
                print(f"{i}. {ex}")

            exercise_choice = input("\nWähle eine Übung (Name oder Nummer): ")

            # Prüfe, ob eine Nummer eingegeben wurde
            try:
                exercise_index = int(exercise_choice) - 1
                if 0 <= exercise_index < len(available_exercises):
                    exercise_name = available_exercises[exercise_index]
                else:
                    print("Ungültige Nummer!")
                    return
            except ValueError:
                # Es wurde ein Name eingegeben
                exercise_name = exercise_choice.lower()
                if exercise_name not in available_exercises:
                    print(f"Übung '{exercise_name}' nicht gefunden!")
                    return

            # Anzahl der Wiederholungen festlegen
            try:
                reps_input = input(f"Wie viele Wiederholungen möchtest du machen? (Standard: {trainer.reps_goal}): ")
                reps_goal = int(reps_input) if reps_input.strip() else trainer.reps_goal
            except ValueError:
                print("Ungültige Eingabe für Wiederholungen. Standardwert wird verwendet.")
                reps_goal = trainer.reps_goal

            # Übung erklären
            if trainer.explain_exercise(exercise_name):
                trainer.start_exercise(exercise_name, reps_goal)

    except KeyboardInterrupt:
        print("\nProgramm vom Benutzer unterbrochen.")
    finally:
        if trainer.exercise_manager:
            trainer.exercise_manager.release()

if __name__ == "__main__":
    main()

