"""
Main Pipeline for Exercise Analysis System
Coordinates all components: Audio, Session, Exercise Data, Backend
"""
import cv2
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

# Local imports from the pipeline directory
from pipeline.audio_system import AudioSystem
from pipeline.session_manager import SessionManager
from pipeline.backend_client import BackendClient
from pipeline.audio_description_manager import AudioDescriptionManager

# Core System Imports
from core.exercise_manager import ExerciseManager
from core.feedback_system import FeedbackType


class ExercisePipeline:
    """Main pipeline for the training system"""

    def __init__(self, mode: str = "development", backend_url: str = "http://localhost:3000"):
        self.mode = mode  # "development" or "production"
        self.backend_url = backend_url

        print(f"Initializing pipeline (Mode: {mode})")

        self.audio_system = AudioSystem()
        self.session_manager = SessionManager(self.audio_system)
        self.audio_description_manager = AudioDescriptionManager()
        self.backend_client = BackendClient(backend_url)
        self.exercise_manager = ExerciseManager("exercises")

        self.camera = None
        self.camera_active = False

        self.current_exercise = None
        self.is_running = False

    def start_pipeline(self):
        """Starts the main pipeline"""
        self.is_running = True

        print("\nWelcome to the Sportsfreund Training System!")

        if self.audio_system:
            self.audio_system.speak(
                "Willkommen beim Sportsfreund Trainingsystem! "
                "Welche Übung möchten Sie heute machen?"
            )

        selected_exercise = self._select_exercise()
        if not selected_exercise:
            print("X --- No exercise selected")
            return

        # Play exercise description
        self._describe_exercise(selected_exercise)

        # Configure session
        sets, reps = self._configure_session()
        if not sets or not reps:
            print("❌ Session configuration aborted")
            return

        # Start session
        session_id = self.session_manager.start_session(selected_exercise, sets, reps)

        # Set exercise in manager
        if not self.exercise_manager.set_exercise(selected_exercise):
            print(f"❌ Could not load exercise '{selected_exercise}'")
            return

        # Start webcam
        if not self._init_camera():
            print("❌ Could not initialize camera")
            return

        try:
            # Run training
            self._run_training_session(sets, reps)

        finally:
            # Cleanup
            self._cleanup()

    def _select_exercise(self) -> Optional[str]:
        """Exercise selection based on mode"""
        available_exercises = self.exercise_manager.get_exercise_list()

        if not available_exercises:
            print("X --- No exercises available")
            return None

        print(f"\nAvailable exercises: {', '.join(available_exercises)}")

        if self.mode == "development":
            return self._select_exercise_text_input(available_exercises)
        else:
            return self._select_exercise_voice_input(available_exercises)

    def _select_exercise_text_input(self, available_exercises: List[str]) -> Optional[str]:
        """Exercise selection via text input (Development Mode)"""
        print("\nDevelopment mode: Text input")

        for i, exercise in enumerate(available_exercises, 1):
            display_name = self.audio_description_manager.get_display_name(exercise)
            print(f"  {i}. {display_name or exercise}")

        try:
            choice = input(f"\nSelect exercise (1-{len(available_exercises)}): ").strip()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_exercises):
                    return available_exercises[idx]
            else:
                for exercise in available_exercises:
                    if choice.lower() in exercise.lower():
                        return exercise

            print("X --- Invalid selection")
            return None

        except KeyboardInterrupt:
            return None

    def _select_exercise_voice_input(self, available_exercises: List[str]) -> Optional[str]:
        """Exercise selection via voice input (Production Mode)"""
        print("\nVoice input")

        if self.audio_system:
            exercise_names = []
            for exercise in available_exercises:
                display_name = self.audio_description_manager.get_display_name(exercise)
                exercise_names.append(display_name or exercise)

            exercises_text = ", ".join(exercise_names)
            self.audio_system.speak(f"Verfügbare Übungen: {exercises_text}. Welche möchten Sie machen?")

        for attempt in range(3):
            print(f"🎤 Speak the exercise name (Attempt {attempt + 1}/3)...")

            spoken_text = self.audio_system.listen_for_command(timeout=10) if self.audio_system else None

            if spoken_text:
                words = spoken_text.lower().split()

                for exercise in available_exercises:
                    exercise_lower = exercise.lower()
                    display_name = self.audio_description_manager.get_display_name(exercise)
                    display_lower = display_name.lower() if display_name else exercise_lower

                    if any(word in exercise_lower or word in display_lower for word in words):
                        print(f"Exercise recognized: {display_name or exercise}")

                        if self.audio_system:
                            self.audio_system.speak(f"{display_name or exercise} ausgewählt. Ist das korrekt?")

                        confirmation = input("Confirm? (y/n): ").strip().lower()
                        if confirmation in ['j', 'ja', 'y', 'yes', '']:
                            return exercise

                print(f"Exercise '{spoken_text}' not recognized")
            else:
                print("No voice input detected")

        print("Voice input failed, fallback to text input")
        return self._select_exercise_text_input(available_exercises)

    def _describe_exercise(self, exercise_name: str):
        """Plays the exercise description"""
        display_name = self.audio_description_manager.get_display_name(exercise_name)
        welcome_text = self.audio_description_manager.get_welcome_text(exercise_name)
        description = self.audio_description_manager.get_description(exercise_name)
        instructions = self.audio_description_manager.get_instructions(exercise_name)

        print(f"\n{display_name or exercise_name}")
        if description:
            print(f"Description: {description}")
        if instructions:
            print(f"Instructions: {instructions}")

        if self.audio_system:
            if welcome_text:
                self.audio_system.speak(welcome_text)
                time.sleep(1)
            if description:
                self.audio_system.speak(description)
                time.sleep(1)
            if instructions:
                self.audio_system.speak(f"Ausführung: {instructions}")

    def _configure_session(self) -> tuple[int, int]:
        """Configures the training session"""
        print("\n⚙Session configuration")

        if self.audio_system:
            self.audio_system.speak("Wie viele Sets möchten Sie machen?")

        if self.mode == "development":
            try:
                sets = int(input("Number of sets (1-10): "))
                if not 1 <= sets <= 10:
                    sets = 3
            except ValueError:
                sets = 3
        else:
            sets = self._get_number_voice_input("Sets", 1, 10, default=3)

        if self.audio_system:
            self.audio_system.speak(f"{sets} Sets. Wie viele Wiederholungen pro Set?")

        if self.mode == "development":
            try:
                reps = int(input("Repetitions per set (1-50): "))
                if not 1 <= reps <= 50:
                    reps = 10
            except ValueError:
                reps = 10
        else:
            reps = self._get_number_voice_input("Repetitions", 1, 50, default=10)

        print(f"Configuration: {sets} sets with {reps} repetitions each")

        if self.audio_system:
            self.audio_system.speak(f"Perfekt! {sets} Sets mit jeweils {reps} Wiederholungen.")

        return sets, reps

    def _get_number_voice_input(self, label: str, min_val: int, max_val: int, default: int) -> int:
        """Gets a number via voice input"""
        for attempt in range(3):
            spoken_text = self.audio_system.listen_for_command(timeout=5) if self.audio_system else None

            if spoken_text:
                words = spoken_text.split()
                for word in words:
                    try:
                        number = int(word)
                        if min_val <= number <= max_val:
                            return number
                    except ValueError:
                        continue

                number_words = {
                    'eins': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 'fünf': 5,
                    'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
                    'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
                    'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50
                }

                for word in words:
                    if word in number_words:
                        number = number_words[word]
                        if min_val <= number <= max_val:
                            return number

        print(f"No valid number recognized, using default: {default}")
        return default

    def _init_camera(self) -> bool:
        """Initializes the webcam"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            self.camera_active = True
            print("✅ Webcam successfully initialized")
            return True

        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False

    def _run_training_session(self, total_sets: int, reps_per_set: int):
        """Runs the complete training session"""
        print(f"\n🏋️ Training starts: {total_sets} sets with {reps_per_set} repetitions each")

        if self.audio_system:
            self.audio_system.speak("Training beginnt in 3 Sekunden. Machen Sie sich bereit!")

        time.sleep(3)

        for set_number in range(1, total_sets + 1):
            print(f"\n🔥 Set {set_number}/{total_sets}")

            self.session_manager.start_set(set_number)

            self._run_set_analysis(set_number, reps_per_set)

            set_summary = self.session_manager.finish_set()

            if set_number < total_sets:
                self._rest_between_sets(set_number, total_sets)

        session_data = self.session_manager.finish_session()

        self._send_to_backend(session_data)

    def _run_set_analysis(self, set_number: int, target_reps: int):
        """Runs live analysis for one set"""
        print(f"📹 Live analysis for set {set_number} starting...")

        if not self.camera or not self.exercise_manager:
            print("❌ Camera or Exercise Manager not available")
            return

        self.exercise_manager.set_feedback_callbacks(
            custom_callback=self._handle_feedback
        )

        frame_count = 0
        start_time = time.time()
        current_reps = 0

        while current_reps < target_reps and self.camera_active:
            ret, frame = self.camera.read()
            if not ret:
                break

            frame_count += 1

            try:
                analysis_result = self.exercise_manager.process_frame(frame)

                status = self.exercise_manager.get_current_status()
                if status and 'reps' in status:
                    new_reps = status['reps']
                    if new_reps > current_reps:
                        current_reps = new_reps
                        self.session_manager.update_rep_count(current_reps)
                        print(f"Repetition {current_reps}/{target_reps}")

                self._draw_live_overlay(frame, current_reps, target_reps, set_number)

                cv2.imshow('Sportsfreund Training', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                continue

        print(f"YEAH - Set {set_number} completed: {current_reps}/{target_reps} repetitions")

    def _handle_feedback(self, feedback_message: str, feedback_type: str = "info"):
        """Handles feedback from Exercise Manager"""
        self.session_manager.add_feedback(
            feedback_type=feedback_type,
            message=feedback_message,
            severity=feedback_type.lower()
        )

        if feedback_type.lower() in ['error', 'safety']:
            if self.audio_system:
                self.audio_system.speak(feedback_message, async_play=True)

    def _draw_live_overlay(self, frame, current_reps: int, target_reps: int, set_number: int):
        """Draws live information on the frame"""
        height, width = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"Set: {set_number}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Repetitions: {current_reps}/{target_reps}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        progress = current_reps / target_reps if target_reps > 0 else 0
        bar_width = 200
        bar_height = 10
        cv2.rectangle(frame, (20, 85), (20 + bar_width, 85 + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 85), (20 + int(bar_width * progress), 85 + bar_height), (0, 255, 0), -1)

        cv2.putText(frame, "ESC to quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _rest_between_sets(self, completed_set: int, total_sets: int):
        """Rest between sets"""
        rest_time = 60

        print(f"⏸️ Break: {rest_time} seconds until set {completed_set + 1}")

        if self.audio_system:
            self.audio_system.speak(f"Pause! {rest_time} Sekunden bis zum nächsten Set.")

        for remaining in range(rest_time, 0, -10):
            if remaining <= 10:
                print(f"⏰ {remaining} seconds...")
                if self.audio_system and remaining <= 5:
                    self.audio_system.speak(str(remaining))
            time.sleep(min(10, remaining))

        if self.audio_system:
            self.audio_system.speak("Pause beendet! Nächster Set beginnt jetzt.")

    def _send_to_backend(self, session_data):
        """Sends session data to backend"""
        print("\nSending data to backend...")

        backend_data = self.session_manager.get_session_data_for_backend()

        success = self.backend_client.send_session_data(backend_data)

        if not success:
            print("💾 Backend not reachable - saving locally")
            self.backend_client.save_session_locally(backend_data)

    def _cleanup(self):
        """Cleans up resources"""
        print("\n🧹 Cleaning up resources...")

        self.camera_active = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        if self.audio_system:
            self.audio_system.cleanup()

        print("Cleanup completed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Sportsfreund Exercise Pipeline')
    parser.add_argument('--mode', choices=['development', 'production'],
                       default='development', help='Execution mode')
    parser.add_argument('--backend-url', default='http://localhost:3000',
                       help='Backend URL')

    args = parser.parse_args()
    pipeline = ExercisePipeline(mode=args.mode, backend_url=args.backend_url)

    try:
        pipeline.start_pipeline()
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline terminated by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        pipeline._cleanup()


if __name__ == "__main__":
    main()
