"""
Audio Description Manager
Manages German audio descriptions separated from technical exercise configs
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class AudioDescriptionManager:
    """Manages audio descriptions for exercises separated from technical configs"""

    def __init__(self, descriptions_dir: str = "audio_descriptions"):
        self.descriptions_dir = Path(descriptions_dir)
        self.descriptions: Dict[str, Dict] = {}
        self._load_descriptions()

    def _load_descriptions(self):
        """Loads all audio descriptions"""
        if not self.descriptions_dir.exists():
            print(f"Audio description directory {self.descriptions_dir} does not exist")
            self._create_default_descriptions()
            return

        for desc_file in self.descriptions_dir.glob("*.json"):
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    desc_data = json.load(f)
                    exercise_name = desc_data.get('exercise_name', desc_file.stem)
                    self.descriptions[exercise_name.lower()] = desc_data
                    print(f"✅ Audio description loaded: {exercise_name}")
            except Exception as e:
                print(f"❌ Error loading {desc_file}: {e}")

    def _create_default_descriptions(self):
        """Creates default audio descriptions and writes them to JSON files"""
        self.descriptions_dir.mkdir(exist_ok=True)
        print("📝 Creating default audio descriptions as JSON files...")

        self._load_descriptions()

    def get_available_exercises(self) -> List[str]:
        """Returns all available exercises with audio descriptions"""
        return list(self.descriptions.keys())

    def get_display_name(self, exercise_name: str) -> str:
        """Returns the German display name"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('display_name_de', exercise_name) if desc else exercise_name

    def get_welcome_text(self, exercise_name: str) -> Optional[str]:
        """Returns the welcome text"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('welcome_text_de') if desc else None

    def get_description(self, exercise_name: str) -> Optional[str]:
        """Returns the description"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('description_de') if desc else None

    def get_instructions(self, exercise_name: str) -> Optional[str]:
        """Returns the execution instructions"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('instructions_de') if desc else None

    def get_preparation_text(self, exercise_name: str) -> Optional[str]:
        """Returns the preparation text"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('preparation_de') if desc else None

    def get_form_cues(self, exercise_name: str) -> List[str]:
        """Returns form hints"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('form_cues_de', []) if desc else []

    def get_motivation_phrases(self, exercise_name: str) -> List[str]:
        """Returns motivation phrases"""
        desc = self.descriptions.get(exercise_name.lower())
        return desc.get('motivation_de', []) if desc else []

    def get_correction_text(self, exercise_name: str, correction_type: str) -> Optional[str]:
        """Returns specific correction text"""
        desc = self.descriptions.get(exercise_name.lower())
        if desc and 'corrections_de' in desc:
            return desc['corrections_de'].get(correction_type)
        return None

    def get_completion_text(self, exercise_name: str, completion_type: str = "set_finished") -> Optional[str]:
        """Returns completion text"""
        desc = self.descriptions.get(exercise_name.lower())
        if desc and 'completion_de' in desc:
            return desc['completion_de'].get(completion_type)
        return None

    def find_exercise_by_keywords(self, keywords: List[str]) -> Optional[str]:
        """Finds exercise based on keywords"""
        keywords_lower = [k.lower() for k in keywords]

        for exercise_name, desc_data in self.descriptions.items():
            if any(keyword in exercise_name for keyword in keywords_lower):
                return exercise_name
            synonyms = desc_data.get('synonyms_de', [])
            for synonym in synonyms:
                if any(keyword in synonym.lower() for keyword in keywords_lower):
                    return exercise_name
            exercise_keywords = desc_data.get('keywords_de', [])
            for ex_keyword in exercise_keywords:
                if any(keyword in ex_keyword.lower() for keyword in keywords_lower):
                    return exercise_name

        return None

    def get_all_exercise_names_for_speech(self) -> str:
        """Returns all exercise names formatted for speech output"""
        names = [self.get_display_name(ex) for ex in self.get_available_exercises()]
        if len(names) == 0:
            return "No exercises available"
        elif len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return f"{', '.join(names[:-1])} and {names[-1]}"
