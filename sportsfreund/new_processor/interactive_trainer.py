import json
import time
import os
import cv2
import logging
import argparse
import numpy as np
import threading
import pyttsx3
from enum import Enum
from main import ExerciseManager

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
    """
    Interaktiver Trainer für Übungen mit ausführlicher Erklärung und angeleiteter Ausführung.
    """
    def __init__(self, camera_index=0, debug=True):
        """
        Initialisiert den InteractiveTrainer.

        Args:
            camera_index: Index der Kamera
            debug: Debug-Modus aktivieren
        """
        self.camera_index = camera_index
        self.debug = debug
        self.exercise_manager = None
        self.state = WorkoutState.WAITING_FOR_SELECTION
        self.current_exercise = None
        self.countdown_seconds = 5
        self.reps_per_feedback = 5
        self.reps_goal = 10  # Standardwert, kann geändert werden
        self.tolerance = 0.15  # 15% Toleranz für Fehlererkennung

        # Sprachausgabe
        self.speech_engine = pyttsx3.init()
        self.speech_engine.setProperty('rate', 150)  # Etwas langsamer sprechen

        # Lade Übungsanleitungen
        self._load_exercise_instructions()

    def _load_exercise_instructions(self):
        """Lädt Übungsanleitungen aus der JSON-Datei"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(base_dir, 'exercise_instructions.json')

        try:
            with open(instructions_path, 'r', encoding='utf-8') as file:
                self.exercise_instructions = json.load(file)
            logging.info(f"Übungsanleitungen geladen. Verfügbare Übungen: {', '.join(self.exercise_instructions.keys())}")
        except Exception as e:
            logging.error(f"Fehler beim Laden der Übungsanleitungen: {e}")
            self.exercise_instructions = {}

    def _speak(self, text, wait=True):
        """Aussprache eines Textes"""


    def list_exercises(self):
        """Zeigt eine Liste aller verfügbaren Übungen an"""
        print("\nVerfügbare Übungen:")
        for i, exercise in enumerate(self.exercise_instructions.keys(), 1):
            print(f"{i}. {exercise}")

        return list(self.exercise_instructions.keys())

    def explain_exercise(self, exercise_name):
        """Erklärt eine Übung ausführlich"""
        if exercise_name not in self.exercise_instructions:
            print(f"Übung '{exercise_name}' nicht gefunden!")
            return False

        self.state = WorkoutState.EXPLAINING
        print(f"\n===== Erklärung: {exercise_name.upper()} =====\n")

        instructions = self.exercise_instructions[exercise_name]

        print("Vorbereitung:")
        self._speak(f"Vorbereitung für {exercise_name}")
        for i, instruction in enumerate(instructions["preparation_instructions"], 1):
            print(f"{i}. {instruction}")
            self._speak(instruction)
            time.sleep(0.1)

        print("\nAusführung:")
        self._speak("Ausführung der Übung")
        for i, instruction in enumerate(instructions["execution_instructions"], 1):
            print(f"{i}. {instruction}")
            self._speak(instruction)
            time.sleep(0.1)

        return True

    def start_countdown(self):
        """Startet einen Countdown vor Beginn der Übung"""
        self.state = WorkoutState.COUNTDOWN
        print("\nBereite dich vor...")
        self._speak("Bereite dich vor. Der Countdown beginnt.")


        print("LOS!")
        self._speak("Los! Starte die Übung.", wait=False)
        self.state = WorkoutState.EXERCISING

    def _generate_exercise_cues(self, exercise_name, phase):
        """Generiert passende Ansagen für jede Phase der Übung"""
        cues = {
            "squats": {
                "down": ["Jetzt nach unten gehen", "In die Hocke gehen", "Runter"],
                "up": ["Wieder hochkommen", "Aufrichten", "Nach oben drücken", "Hoch"],
                "bottom": ["Halte diese Position", "Halte die tiefe Position"]
            },
            "push_ups": {
                "down": ["Senke dich nach unten", "Brust Richtung Boden", "Runter"],
                "up": ["Drücke dich hoch", "Nach oben drücken", "Hoch"],
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
        config_path = os.path.join(base_dir, exercise_name, 'config.json')

        self.exercise_manager = ExerciseManager(
            exercise_name=exercise_name,
            config_path=config_path,
            debug=self.debug
        )

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
                cv2.imshow('Interaktives Training', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            coords = self.exercise_manager.pose_extractor.get_landmark_coordinates(landmarks, width, height)
            joint_angles = self.exercise_manager.pose_extractor.calculate_joint_angles(coords)

            # Verarbeite Frame mit dem Übungs-Analysator
            analyzed_frame, status = self.exercise_manager.analyzer.analyze_frame(processed_frame, joint_angles, coords, self.debug)

            # Aktuelle Wiederholungszahl und Ziel anzeigen
            cv2.putText(analyzed_frame, f"Wiederholung: {self.exercise_manager.analyzer.rep_count}/{self.reps_goal}",
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Sprachansagen für Phasenwechsel
            current_phase = self.exercise_manager.analyzer.current_phase
            if current_phase != self.last_phase and current_phase is not None:
                cue = self._generate_exercise_cues(exercise_name, current_phase)
                self._speak(cue, wait=False)
                self.last_phase = current_phase

            # Rep-Counter prüfen
            if self.exercise_manager.analyzer.rep_count > self.current_rep:
                self.current_rep = self.exercise_manager.analyzer.rep_count
                print(f"Wiederholung {self.current_rep} abgeschlossen!")

                # Bei jeder 5. Wiederholung Feedback geben
                if self.current_rep % self.reps_per_feedback == 0 and self.current_rep > self.last_feedback_rep:
                    self._give_feedback()
                    self.last_feedback_rep = self.current_rep

                # Ziel erreicht?
                if self.current_rep >= self.reps_goal:
                    self._speak("Gut gemacht! Du hast dein Ziel erreicht.", wait=False)
                    print("\nZiel erreicht! Super gemacht!")
                    time.sleep(3)  # Zeit zum Feiern
                    break

            cv2.imshow('Interaktives Training', analyzed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\nÜbung beendet.")
        print(f"Ausgeführte Wiederholungen: {self.exercise_manager.analyzer.rep_count}")

        self.state = WorkoutState.COMPLETED
        return self.exercise_manager.analyzer.rep_count

    def _give_feedback(self):
        """Gibt Feedback zur Übungsausführung basierend auf der Feedback-Historie"""
        self.state = WorkoutState.FEEDBACK

        # Analysiere die bisherige Feedback-Historie
        feedback_history = self.exercise_manager.analyzer.feedback_history

        if not feedback_history:
            self._speak("Gute Arbeit! Mach weiter so.", wait=False)
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
                self._speak(feedback_text, wait=False)
                return

        # Wenn kein spezifisches Problem erkannt wurde oder die Ausführung gut ist
        self._speak("Gute Ausführung! Weiter so.", wait=False)

    def start_interactive_session(self):
        """Startet eine interaktive Trainingseinheit"""
        print("\n=== Willkommen beim interaktiven Fitnesstrainer! ===\n")
        self._speak("Willkommen beim interaktiven Fitnesstrainer!")

        while True:
            if self.state == WorkoutState.WAITING_FOR_SELECTION:
                print("\nWähle eine Option:")
                print("1. Übung auswählen")
                print("2. Beenden")

                choice = input("Deine Wahl (1-2): ")

                if choice == '1':
                    available_exercises = self.list_exercises()

                    exercise_choice = input("\nWähle eine Übung (Name oder Nummer): ")

                    # Prüfe, ob eine Nummer eingegeben wurde
                    try:
                        exercise_index = int(exercise_choice) - 1
                        if 0 <= exercise_index < len(available_exercises):
                            exercise_name = available_exercises[exercise_index]
                        else:
                            print("Ungültige Nummer!")
                            continue
                    except ValueError:
                        # Es wurde ein Name eingegeben
                        exercise_name = exercise_choice.lower()
                        if exercise_name not in available_exercises:
                            print(f"Übung '{exercise_name}' nicht gefunden!")
                            continue

                    # Anzahl der Wiederholungen festlegen
                    try:
                        reps_input = input(f"Wie viele Wiederholungen möchtest du machen? (Standard: {self.reps_goal}): ")
                        reps_goal = int(reps_input) if reps_input.strip() else self.reps_goal
                    except ValueError:
                        print("Ungültige Eingabe für Wiederholungen. Standardwert wird verwendet.")
                        reps_goal = self.reps_goal

                    # Übung erklären
                    if self.explain_exercise(exercise_name):
                        input("\nDrücke ENTER, wenn du bereit bist zu beginnen...")
                        self.start_countdown()
                        self.start_exercise(exercise_name, reps_goal)

                elif choice == '2':
                    print("\nTraining beendet. Bis zum nächsten Mal!")
                    self._speak("Training beendet. Bis zum nächsten Mal!")
                    break

                else:
                    print("Ungültige Eingabe. Bitte wähle 1 oder 2.")

            # Nach Abschluss einer Übung zurück zur Auswahl
            elif self.state == WorkoutState.COMPLETED:
                self.state = WorkoutState.WAITING_FOR_SELECTION
                input("\nDrücke ENTER, um fortzufahren...")

        # Ressourcen freigeben
        if self.exercise_manager:
            self.exercise_manager.release()
        self.speech_engine.stop()

def main():
    """
    Hauptfunktion für die Kommandozeilenausführung.
    """
    parser = argparse.ArgumentParser(description='Interaktiver Fitness-Trainer')

    parser.add_argument('--camera', type=int, default=0,
                        help='Index der Webcam (0 für Standardkamera)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug-Modus aktivieren')

    args = parser.parse_args()

    trainer = InteractiveTrainer(
        camera_index=args.camera,
        debug=args.debug
    )

    try:
        trainer.start_interactive_session()
    except KeyboardInterrupt:
        print("\nProgramm vom Benutzer unterbrochen.")
    finally:
        if trainer.exercise_manager:
            trainer.exercise_manager.release()

if __name__ == "__main__":
    main()
