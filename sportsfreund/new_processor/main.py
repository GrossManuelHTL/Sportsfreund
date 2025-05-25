import os
import cv2
import time
import logging
import argparse
from poseextraction.pose_extractor import PoseExtractor
from exercise_config import ExerciseConfig
from analyzers.squat_analyzer import SquatAnalyzer

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExerciseManager:
    """
    Hauptklasse für die Verwaltung und Durchführung von Übungsanalysen.
    """
    def __init__(self, exercise_name=None, config_path=None, debug=True):
        """
        Initialisiert den ExerciseManager.

        Args:
            exercise_name: Name der Übung
            config_path: Optionaler Pfad zur Konfigurationsdatei
            debug: Debug-Modus aktivieren
        """
        self.debug = debug
        self.exercise_name = exercise_name

        if config_path is None and exercise_name is not None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'exercises', exercise_name, 'config.json')

        self.config = ExerciseConfig(config_path)

        if not self.exercise_name:
            self.exercise_name = self.config.get_exercise_name()

        # Initialisiere Pose-Extraktor
        self.pose_extractor = PoseExtractor()

        # Initialisiere Übungs-Analysator
        self.analyzer = SquatAnalyzer(self.config)
        self.feedback_map = self.analyzer.feedback_map

        logging.info(f"ExerciseManager für {self.exercise_name} initialisiert")

    def process_video(self, video_path, output_path=None, show_video=True, flip=False):
        """
        Verarbeitet ein Übungsvideo.

        Args:
            video_path: Pfad zum Eingabevideo
            output_path: Optional, Pfad für das Ausgabevideo
            show_video: Video während der Verarbeitung anzeigen
            flip: Video horizontal spiegeln

        Returns:
            dict: Analyseergebnisse
        """
        # Öffne das Video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_ZOOM, 0);
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0);

        # Prüfe, ob das Video geöffnet werden konnte
        if not cap.isOpened():
            logging.error(f"Fehler beim Öffnen des Videos: {video_path}")
            return None

        # Hole Video-Eigenschaften
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Vorbereitung für Ausgabevideo
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Leistungs-Tracking
        start_time = time.time()
        frame_number = 0
        processed_frames = 0

        logging.info(f"Starte Verarbeitung von {video_path}")
        logging.info(f"Video-Eigenschaften: {width}x{height}, {fps} FPS, {frame_count} Frames")

        # Übungsanalyse zurücksetzen
        self.analyzer.reset()

        # Verarbeite jeden Frame
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Frame-Nummer
            frame_number += 1

            # Frame spiegeln, wenn gewünscht
            if flip:
                frame = cv2.flip(frame, 1)

            # Frame mit dem Pose-Extraktor verarbeiten
            landmarks, processed_frame = self.pose_extractor.extract_landmarks(frame)

            # Wenn keine Landmarken erkannt wurden
            if landmarks is None:
                if self.debug:
                    print(f"Keine Pose erkannt in Frame {frame_number}")

                # Originalframe zum Ausgabevideo hinzufügen
                if out:
                    out.write(frame)

                # Video anzeigen
                if show_video:
                    cv2.imshow('Exercise Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                continue

            coords = self.pose_extractor.get_landmark_coordinates(landmarks, width, height)
            joint_angles = self.pose_extractor.calculate_joint_angles(coords)

            analyzed_frame, status, feedback_keys = self.analyzer.analyze_frame(processed_frame, joint_angles, coords, self.debug)

            if self.debug and frame_number % 30 == 0:
                print(f"Frame {frame_number}/{frame_count} ({frame_number/frame_count*100:.1f}%)")
                print(f"Status: {status}")

                if joint_angles:
                    print("Winkel:")
                    for name, angle in joint_angles.items():
                        print(f"  {name}: {angle:.1f}°")
                print("-" * 40)

            processed_frames += 1

            if out:
                out.write(analyzed_frame)

            if show_video:
                cv2.imshow('Exercise Analysis', analyzed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        elapsed_time = time.time() - start_time
        time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0

        logging.info(f"Videoanalyse abgeschlossen.")
        logging.info(f"Verarbeitet: {processed_frames}/{frame_count} Frames")
        logging.info(f"Gesamtzeit: {elapsed_time:.2f} Sekunden")
        logging.info(f"Zeit pro Frame: {time_per_frame*1000:.2f} ms")
        logging.info(f"Wiederholungen: {self.analyzer.rep_count}")

        result = self.analyzer._get_status()
        result['time_per_frame_ms'] = time_per_frame * 1000
        result['total_time_seconds'] = elapsed_time
        result['processed_frames'] = processed_frames
        result['total_frames'] = frame_count

        return result

    def process_webcam(self, camera_index=0, output_path=None, flip=False):
        """
        Verarbeitet den Videostream einer Webcam.

        Args:
            camera_index: Index der Kamera
            output_path: Optional, Pfad für das Ausgabevideo
            flip: Video horizontal spiegeln

        Returns:
            dict: Analyseergebnisse
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            logging.error(f"Fehler beim Öffnen der Webcam mit Index {camera_index}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Annahme für Webcam

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_time = time.time()
        frame_number = 0
        fps_time = 0

        logging.info(f"Starte Webcam-Verarbeitung")
        logging.info(f"Video-Eigenschaften: {width}x{height}")

        self.analyzer.reset()

        print("Drücken Sie 'q' zum Beenden")
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Frame-Nummer
            frame_number += 1

            if flip:
                frame = cv2.flip(frame, 1)

            elapsed_time = time.time() - start_time
            fps = frame_number / elapsed_time

            landmarks, processed_frame = self.pose_extractor.extract_landmarks(frame)

            if landmarks is None:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if out:
                    out.write(frame)

                cv2.imshow('Exercise Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                continue

            coords = self.pose_extractor.get_landmark_coordinates(landmarks, width, height)
            joint_angles = self.pose_extractor.calculate_joint_angles(coords)

            analyzed_frame, status, feedback_keys = self.analyzer.analyze_frame(processed_frame, joint_angles, coords, self.debug)

            cv2.putText(analyzed_frame, f"FPS: {fps:.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if self.debug and frame_number % 30 == 0:
                print(f"Frame {frame_number}, FPS: {fps:.1f}")
                print(f"Status: {status}")

                if joint_angles:
                    print("Winkel:")
                    for name, angle in joint_angles.items():
                        print(f"  {name}: {angle:.1f}°")
                print("-" * 40)

            # Zum Ausgabevideo hinzufügen
            if out:
                out.write(analyzed_frame)

            cv2.imshow('Exercise Analysis', analyzed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        elapsed_time = time.time() - start_time
        logging.info(f"Webcam-Analyse beendet.")
        logging.info(f"Gesamtzeit: {elapsed_time:.2f} Sekunden")
        logging.info(f"Verarbeitete Frames: {frame_number}")
        logging.info(f"Durchschnittliche FPS: {frame_number/elapsed_time:.1f}")
        logging.info(f"Wiederholungen: {self.analyzer.rep_count}")

        result = self.analyzer._get_status()
        result['average_fps'] = frame_number / elapsed_time
        result['total_time_seconds'] = elapsed_time
        result['processed_frames'] = frame_number

        return result

    def release(self):
        """
        Gibt alle Ressourcen frei.
        """
        if hasattr(self, 'pose_extractor'):
            self.pose_extractor.release()

def list_exercises():
    """
    Listet alle verfügbaren Übungen aus dem exercises-Ordner auf.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exercises_dir = os.path.join(base_dir, 'exercises')

    available_exercises = []
    for item in os.listdir(exercises_dir):
        item_path = os.path.join(exercises_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
            available_exercises.append(item)

    return available_exercises

def main():
    """
    Hauptfunktion für die Kommandozeilenausführung.
    """
    parser = argparse.ArgumentParser(description='Fitness-Übungsanalysator')

    parser.add_argument('--exercise', type=str,
                        help='Name der zu analysierenden Übung')

    # Hauptmodus-Gruppe
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--video', type=str,
                          help='Pfad zum Eingabevideo')
    mode_group.add_argument('--webcam', action='store_true',
                          help='Webcam-Modus verwenden')
    mode_group.add_argument('--interactive', action='store_true',
                          help='Interaktiver Trainingsmodus')

    parser.add_argument('--camera', type=int, default=0,
                        help='Index der Webcam (0 für Standardkamera)')
    parser.add_argument('--output', type=str,
                        help='Pfad für das Ausgabevideo')
    parser.add_argument('--config', type=str,
                        help='Pfad zur Konfigurationsdatei')
    parser.add_argument('--no-display', action='store_true',
                        help='Video während der Verarbeitung nicht anzeigen')
    parser.add_argument('--flip', action='store_true',
                        help='Video horizontal spiegeln')
    parser.add_argument('--debug', action='store_true',
                        help='Debug-Modus aktivieren')

    args = parser.parse_args()

    if args.interactive:
        # Interaktiver Trainingsmodus
        from interactive_trainer import InteractiveTrainer

        print("\n=== Willkommen beim Fitness-Übungsanalysator - Interaktiver Modus ===\n")

        trainer = InteractiveTrainer(
            exercise=args.exercise,
            camera_index=args.camera,
            debug=args.debug
        )

        try:
            if args.exercise:
                available_exercises = trainer.list_exercises()
                if args.exercise in available_exercises:
                    print(f"Starte direkt mit Übung: {args.exercise}")

                    # Anzahl der Wiederholungen
                    reps_goal = int(input(f"Wie viele Wiederholungen möchtest du machen? (Standard: {trainer.reps_goal}): ") or trainer.reps_goal)

                    trainer.explain_exercise(args.exercise)
                    input("\nDrücke ENTER, wenn du bereit bist zu beginnen...")
                    trainer.start_exercise(args.exercise, reps_goal)
                else:
                    print(f"Übung '{args.exercise}' nicht gefunden!")
                    print("Verfügbare Übungen:")
                    for exercise in available_exercises:
                        print(f" - {exercise}")
            else:
                # Übungsauswahl
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

                try:
                    reps_input = input(f"Wie viele Wiederholungen möchtest du machen? (Standard: {trainer.reps_goal}): ")
                    reps_goal = int(reps_input) if reps_input.strip() else trainer.reps_goal
                except ValueError:
                    print("Ungültige Eingabe für Wiederholungen. Standardwert wird verwendet.")
                    reps_goal = trainer.reps_goal

                if trainer.explain_exercise(exercise_name):
                    input("\nDrücke ENTER, wenn du bereit bist zu beginnen...")
                    trainer.start_exercise(exercise_name, reps_goal)

        except KeyboardInterrupt:
            print("\nProgramm vom Benutzer unterbrochen.")
        finally:
            if trainer.exercise_manager:
                trainer.exercise_manager.release()

    else:
        # Wenn keine Übung angegeben wurde und wir nicht im interaktiven Modus sind
        if not args.exercise:
            print("Verfügbare Übungen:")
            for exercise in list_exercises():
                print(f" - {exercise}")
            print("\nBitte --exercise ÜBUNGSNAME angeben.")
            return

        # Normaler Analyse-Modus
        exercise_manager = ExerciseManager(
            exercise_name=args.exercise,
            config_path=args.config,
            debug=args.debug
        )

        try:
            if args.video:
                # Video-Modus
                result = exercise_manager.process_video(
                    args.video,
                    output_path=args.output,
                    show_video=not args.no_display,
                    flip=args.flip
                )
            elif args.webcam:
                # Webcam-Modus
                result = exercise_manager.process_webcam(
                    camera_index=args.camera,
                    output_path=args.output,
                    flip=args.flip
                )

            if result:
                print("\nAnalyse-Ergebnisse:")
                print(f"Übung: {result['exercise']}")
                print(f"Wiederholungen: {result['rep_count']}")
                print(f"Aktuelle Phase: {result['current_phase']}")
                print(f"Zeit: {result['time_elapsed']:.2f} Sekunden")

        finally:
            exercise_manager.release()


if __name__ == "__main__":
    main()

