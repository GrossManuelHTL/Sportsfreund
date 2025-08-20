"""
Exercise Data System
Manages exercise configurations and audio descriptions separately
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from .audio_description_manager import AudioDescriptionManager


class ExerciseDataManager:
    """Manages technical exercise data and audio descriptions separately"""

    def __init__(self, exercises_dir: str = "exercises", audio_descriptions_dir: str = "audio_descriptions"):
        self.exercises_dir = Path(exercises_dir)
        self.exercise_configs: Dict[str, Dict] = {}
        self.audio_manager = AudioDescriptionManager(audio_descriptions_dir)
        self._load_exercise_configs()

    def _load_exercise_configs(self):
        """Loads all technical exercise configurations"""
        if not self.exercises_dir.exists():
            print(f"Exercise directory {self.exercises_dir} does not exist")
            return

        for exercise_file in self.exercises_dir.glob("*.json"):
            try:
                with open(exercise_file, 'r', encoding='utf-8') as f:
                    exercise_data = json.load(f)
                    exercise_name = exercise_data.get('name', exercise_file.stem)
                    self.exercise_configs[exercise_name.lower()] = exercise_data
                    print(f"✅ Exercise config loaded: {exercise_name}")
            except Exception as e:
                print(f"❌ Error loading {exercise_file}: {e}")

    def get_available_exercises(self) -> List[str]:
        """Returns a list of all available exercises (from both systems)"""
        # Combine exercises from both systems
        config_exercises = set(self.exercise_configs.keys())
        audio_exercises = set(self.audio_manager.get_available_exercises())

        # Prefer exercises that exist in both systems
        combined_exercises = config_exercises.union(audio_exercises)
        return list(combined_exercises)

    def get_exercise_config(self, exercise_name: str) -> Optional[Dict]:
        """Returns the technical configuration of an exercise"""
        return self.exercise_configs.get(exercise_name.lower())

    def has_exercise_config(self, exercise_name: str) -> bool:
        """Checks if a technical configuration exists"""
        return exercise_name.lower() in self.exercise_configs

    def has_audio_description(self, exercise_name: str) -> bool:
        """Checks if an audio description exists"""
        return exercise_name.lower() in self.audio_manager.get_available_exercises()

    # Audio-related methods delegate to AudioDescriptionManager
    def get_display_name(self, exercise_name: str) -> str:
        """Returns the German display name"""
        return self.audio_manager.get_display_name(exercise_name)

    def get_welcome_text(self, exercise_name: str) -> Optional[str]:
        """Returns the welcome text"""
        return self.audio_manager.get_welcome_text(exercise_name)

    def get_description(self, exercise_name: str) -> Optional[str]:
        """Returns the audio description"""
        return self.audio_manager.get_description(exercise_name)

    def get_instructions(self, exercise_name: str) -> Optional[str]:
        """Returns the audio instructions"""
        return self.audio_manager.get_instructions(exercise_name)

    def get_preparation_text(self, exercise_name: str) -> Optional[str]:
        """Returns the preparation text"""
        return self.audio_manager.get_preparation_text(exercise_name)

    def get_form_cues(self, exercise_name: str) -> List[str]:
        """Returns form hints"""
        return self.audio_manager.get_form_cues(exercise_name)

    def get_motivation_phrases(self, exercise_name: str) -> List[str]:
        """Returns motivation phrases"""
        return self.audio_manager.get_motivation_phrases(exercise_name)

    def get_correction_text(self, exercise_name: str, correction_type: str) -> Optional[str]:
        """Returns specific correction text"""
        return self.audio_manager.get_correction_text(exercise_name, correction_type)

    def get_completion_text(self, exercise_name: str, completion_type: str = "set_finished") -> Optional[str]:
        """Returns completion text"""
        return self.audio_manager.get_completion_text(exercise_name, completion_type)

    def find_exercise_by_keywords(self, keywords: List[str]) -> Optional[str]:
        """Finds an exercise based on keywords"""
        return self.audio_manager.find_exercise_by_keywords(keywords)

    def get_all_exercise_names_for_speech(self) -> str:
        """Returns all exercise names formatted for speech output"""
        return self.audio_manager.get_all_exercise_names_for_speech()

    def get_complete_exercise_info(self, exercise_name: str) -> Dict:
        """Returns both technical config and audio info"""
        result = {
            "exercise_name": exercise_name,
            "has_config": self.has_exercise_config(exercise_name),
            "has_audio": self.has_audio_description(exercise_name),
            "config": self.get_exercise_config(exercise_name),
            "display_name": self.get_display_name(exercise_name),
            "description": self.get_description(exercise_name),
            "instructions": self.get_instructions(exercise_name)
        }
        return result


# Create default exercise data
def create_default_exercise_files(exercises_dir: Path):
    """Creates default exercise files"""
    exercises_dir.mkdir(exist_ok=True)

    # Squats
    squats_data = {
        "name": "squats",
        "display_name_de": "Kniebeugen",
        "description_de": "Kniebeugen sind eine grundlegende Übung für die Bein- und Gesäßmuskulatur. Sie stärken Quadrizeps, Gesäßmuskeln und den Rumpf.",
        "instructions_de": "Stellen Sie sich mit den Füßen schulterbreit auseinander. Senken Sie Ihren Körper ab, als würden Sie sich auf einen Stuhl setzen. Halten Sie den Rücken gerade und die Knie über den Zehen. Drücken Sie sich dann wieder nach oben.",
        "synonyms_de": ["kniebeuge", "hocke", "squat"],
        "keywords_de": ["beine", "po", "gesäß", "oberschenkel"],
        "target_muscles_de": ["Quadrizeps", "Gesäßmuskulatur", "Rumpf"],
        "difficulty": "beginner",
        "equipment": "none",
        "safety_tips_de": [
            "Halten Sie die Knie über den Zehen",
            "Vermeiden Sie ein Hohlkreuz",
            "Gehen Sie nur so tief wie es comfortable ist",
            "Halten Sie das Gewicht auf den Fersen"
        ],
        "common_mistakes_de": [
            "Knie fallen nach innen",
            "Rücken wird rund",
            "Nicht tief genug",
            "Gewicht auf den Zehen"
        ],
        "state_machine": {
            "states": ["standing", "descending", "bottom", "ascending"],
            "transitions": {
                "standing": ["descending"],
                "descending": ["bottom", "standing"],
                "bottom": ["ascending"],
                "ascending": ["standing"]
            }
        },
        "pose_requirements": {
            "key_points": ["left_knee", "right_knee", "left_hip", "right_hip", "nose"],
            "angles": {
                "knee_min": 70,
                "knee_max": 170,
                "hip_min": 70
            }
        }
    }

    # Push-ups
    pushups_data = {
        "name": "pushups",
        "display_name_de": "Liegestütze",
        "description_de": "Liegestütze sind eine klassische Übung für die Brust-, Schulter- und Armmuskulatur sowie den Rumpf.",
        "instructions_de": "Beginnen Sie in der Plank-Position mit den Händen unter den Schultern. Senken Sie den Körper ab, bis die Brust fast den Boden berührt. Drücken Sie sich dann wieder nach oben.",
        "synonyms_de": ["liegestütz", "push-up", "pushup"],
        "keywords_de": ["brust", "arme", "schultern", "rumpf"],
        "target_muscles_de": ["Brustmuskulatur", "Trizeps", "Deltamuskeln", "Rumpf"],
        "difficulty": "intermediate",
        "equipment": "none",
        "safety_tips_de": [
            "Halten Sie den Körper in einer geraden Linie",
            "Spannen Sie den Rumpf an",
            "Bewegen Sie sich kontrolliert",
            "Atmung nicht vergessen"
        ],
        "common_mistakes_de": [
            "Hüfte hängt durch",
            "Po zu hoch",
            "Ellbogen zu weit vom Körper",
            "Nicht vollständige Bewegung"
        ]
    }

    # Write files
    with open(exercises_dir / "squats.json", 'w', encoding='utf-8') as f:
        json.dump(squats_data, f, indent=2, ensure_ascii=False)

    with open(exercises_dir / "pushups.json", 'w', encoding='utf-8') as f:
        json.dump(pushups_data, f, indent=2, ensure_ascii=False)

    print("✅ Default exercise files created")


if __name__ == "__main__":
    # Test Exercise Data Manager
    exercises_dir = Path("../exercises")
    create_default_exercise_files(exercises_dir)

    manager = ExerciseDataManager(str(exercises_dir))
    print("Available exercises:", manager.get_available_exercises())
