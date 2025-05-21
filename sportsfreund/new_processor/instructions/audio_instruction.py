import pyttsx3
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

class InstructionExplainer:
    def __init__(self, exercise):
        self.instructions = None
        self.exercise = exercise
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ex_path = os.path.join(base_dir, f"{exercise}_instruction.json")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1)
        self.engine.setProperty('voice', 'english+f3')
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
            self.engine.say(step)
            self.engine.runAndWait()
            logging.info(f"[INFO] Speaking preparation instruction: {step}")
        self.engine.stop()
        
    def speak_exec_instruction(self):
        for i, step in enumerate(self.exec_instruction):
            print(f"Ausführung {i+1}/{len(self.exec_instruction)}: {step}")
            self.engine.say(step)
            self.engine.runAndWait()
            logging.info(f"[INFO] Speaking execution instruction: {step}")
        self.engine.stop()

    def say_sentence(self, sentence, is_last=False):
        print(f"{sentence}")
        self.engine.say(sentence)
        self.engine.runAndWait()
        logging.info(f"[INFO] Speaking sentence: {sentence}")
        if is_last:
            self.engine.stop()
