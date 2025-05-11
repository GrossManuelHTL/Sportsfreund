import pygame
from gtts import gTTS
import tempfile
import os
import time


class SquatAudioInstructions:
    @staticmethod
    def get_preparation_instructions():
        return [
            "Stand with your feet shoulder-width apart.",
            "Point your toes slightly outward.",
            "Keep your chest up and shoulders back.",
            "Look straight ahead to maintain proper neck alignment."
        ]

    @staticmethod
    def get_execution_instructions():
        return [
            "Begin by taking a deep breath in.",
            "Bend at your knees and hips, sitting back as if into a chair.",
            "Keep your knees behind your toes.",
            "Lower until your thighs are at least parallel to the floor.",
            "Keep your back straight throughout the movement.",
            "Push through your heels to return to standing position.",
            "Exhale as you rise back up.",
            "Fully extend your hips at the top of the movement."
        ]

    @staticmethod
    def get_full_audio_instructions():
        instructions = []
        instructions.append("Welcome to the squat exercise guide. Let's begin with proper form.")
        instructions.append("First, get into the starting position:")
        for step in SquatAudioInstructions.get_preparation_instructions():
            instructions.append(step)
        instructions.append("Now, let's perform the squat:")
        for step in SquatAudioInstructions.get_execution_instructions():
            instructions.append(step)
        instructions.append("Remember to maintain proper form throughout each repetition.")
        instructions.append(
            "Start with a comfortable number of repetitions and gradually increase as you build strength.")
        return instructions

    def __init__(self, language="en", slow=False):
        """Initialize TTS parameters"""
        self.language = language
        self.slow = slow
        pygame.mixer.init()
        self.temp_files = []

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
        """Speak all squat instructions with pauses between them"""
        instructions = self.get_full_audio_instructions()

        print("===== COMPLETE SQUAT INSTRUCTIONS =====")
        try:
            for instruction in instructions:
                self.speak_text(instruction)
                time.sleep(pause_between)
        finally:
            pass


if __name__ == "__main__":
    squat_guide = SquatAudioInstructions()

    try:
        squat_guide.speak_all_instructions(pause_between=0.5)

    finally:
        squat_guide.cleanup_temp_files()