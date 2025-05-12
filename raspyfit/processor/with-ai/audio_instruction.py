import pygame
from gtts import gTTS
import tempfile
import os
import time
import json


class ExerciseAudioInstructions:
    def __init__(self, exercise_name, language="en", slow=False, instructions_file="exercise_instructions.json"):
        """Initialize TTS parameters and load exercise instructions"""
        self.exercise_name = exercise_name
        self.language = language
        self.slow = slow
        self.instructions_file = instructions_file
        self.instructions = self._load_instructions()
        pygame.mixer.init()
        self.temp_files = []

    def _load_instructions(self):
        """Load exercise instructions from JSON file"""
        try:
            with open(self.instructions_file, 'r') as file:
                all_instructions = json.load(file)

            if self.exercise_name not in all_instructions:
                raise ValueError(f"Exercise '{self.exercise_name}' not found in instructions file")

            return all_instructions[self.exercise_name]
        except FileNotFoundError:
            raise FileNotFoundError(f"Instructions file '{self.instructions_file}' not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{self.instructions_file}'")

    def get_preparation_instructions(self):
        """Get preparation instructions for the current exercise"""
        return self.instructions.get("preparation_instructions", [])

    def get_execution_instructions(self):
        """Get execution instructions for the current exercise"""
        return self.instructions.get("execution_instructions", [])

    def get_full_audio_instructions(self):
        """Returns complete auditory instructions for the exercise"""
        instructions = []
        instructions.append(f"Welcome to the {self.exercise_name} exercise guide. Let's begin with proper form.")
        instructions.append("First, get into the starting position:")
        for step in self.get_preparation_instructions():
            instructions.append(step)
        instructions.append("Now, let's perform the exercise:")
        for step in self.get_execution_instructions():
            instructions.append(step)
        instructions.append("Remember to maintain proper form throughout each repetition.")
        instructions.append(
            "Start with a comfortable number of repetitions and gradually increase as you build strength.")
        return instructions

    def __del__(self):
        """Clean up temporary files when the object is destroyed"""
        self.cleanup_temp_files()

    def cleanup_temp_files(self):
        """Clean up any remaining temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self.temp_files = []

    def speak_text(self, text):
        """Speak text directly using gTTS and pygame"""
        print(f"• {text}")

        temp_fd, temp_filename = tempfile.mkstemp(suffix='.mp3')
        os.close(temp_fd)
        self.temp_files.append(temp_filename)

        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        tts.save(temp_filename)

        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()

    def speak_all_instructions(self, pause_between=1.0):
        """Speak all exercise instructions with pauses between them"""
        instructions = self.get_full_audio_instructions()

        print(f"===== COMPLETE {self.exercise_name.upper()} INSTRUCTIONS =====")
        try:
            for instruction in instructions:
                self.speak_text(instruction)
                time.sleep(pause_between)
        finally:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exercise Audio Instructions')
    parser.add_argument('exercise', type=str, help='Name of the exercise (e.g., squat)')
    parser.add_argument('--pause', type=float, default=0.5, help='Pause between instructions in seconds')
    parser.add_argument('--language', type=str, default='en', help='Language code for TTS')
    parser.add_argument('--slow', action='store_true', help='Use slower TTS speed')
    parser.add_argument('--instructions-file', type=str, default='exercise_instructions.json',
                        help='Path to instructions JSON file')

    args = parser.parse_args()

    exercise_guide = ExerciseAudioInstructions(
        exercise_name=args.exercise,
        language=args.language,
        slow=args.slow,
        instructions_file=args.instructions_file
    )

    try:
        exercise_guide.speak_all_instructions(pause_between=args.pause)
    finally:
        exercise_guide.cleanup_temp_files()