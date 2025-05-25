import pyttsx3
import json
import logging
import os
import pygame
from gtts import gTTS
import time

logging.basicConfig(level=logging.INFO)

class InstructionExplainer:
    def __init__(self, exercise):
        self.instructions = None
        self.exercise = exercise
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ex_path = os.path.join(base_dir, f"{exercise}_instruction.json")
        self.engine = pygame.mixer
        self.engine.init()
        self.prep_instruction = None
        self.exec_instruction = None
        self.load_instructions()

    def load_instructions(self):
        try:
            with open(self.ex_path, 'r') as f:
                raw_instructions = json.load(f)
                self.instructions = raw_instructions.get(self.exercise, {})
                self.prep_instruction = self.instructions.get("preparation_instructions", [])
                self.exec_instruction = self.instructions.get("execution_instructions", [])
                logging.info(f"[INFO] Instructions loaded for {self.ex_path}")
        except FileNotFoundError:
            print(f"[ERROR] Instruction file not found: {self.ex_path}")
            logging.info(f"[ERROR] Instruction file not found: {self.ex_path}")
            self.instructions = {}

    def speak_prep_instruction(self):
        for i, step in enumerate(self.prep_instruction):
            print(f"Vorbereitung {i+1}/{len(self.prep_instruction)}: {step}")
            self.say_sentence(step)
            logging.info(f"[INFO] Speaking preparation instruction: {step}")
        self.engine.stop()
        
    def speak_exec_instruction(self):
        for i, step in enumerate(self.exec_instruction):
            print(f"Ausführung {i+1}/{len(self.exec_instruction)}: {step}")
            self.say_sentence(step)
            logging.info(f"[INFO] Speaking execution instruction: {step}")
        self.engine.stop()

    def say_sentence(self, sentence):
        print(f"{sentence}")
        tts = gTTS(sentence, lang='de')

        try:
            tts.save("temp.mp3")
        except PermissionError:
            pygame.mixer.music.unload()
            time.sleep(0.1)
            tts.save("temp.mp3")

        logging.info(f"[INFO] Speaking sentence: {sentence}")
        self.engine.music.load("temp.mp3")
        self.engine.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()

    def say_sentence_no_wait(self, sentence):
        print(f"{sentence}")
        tts = gTTS(sentence, lang='de')

        try:
            tts.save("temp.mp3")
        except PermissionError:
            pygame.mixer.music.unload()
            time.sleep(0.1)
            tts.save("temp.mp3")

        logging.info(f"[INFO] Speaking sentence: {sentence}")
        self.engine.music.load("temp.mp3")
        self.engine.music.play()

    def stop_speaking(self):
        self.engine.music.stop()
        logging.info("[INFO] Stopped speaking.")
        pygame.mixer.music.unload()