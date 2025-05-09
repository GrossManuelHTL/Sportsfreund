import json
import os


class ExerciseManager:
    """
    Manage exercises and their configurations.
    All exercises are stored in a directory called "exercises" in the root directory.

    """

    def __init__(self, exercises_dir="exercises"):
        """
        Initializes the ExerciseManager with the specified exercises directory.

        Args:
            exercises_dir: Directory path for the exercises. Default is "exercises" in the root directory.
        """
        self.exercises_dir = exercises_dir

        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(exercises_dir, exist_ok=True)

    def get_available_exercises(self):
        """
        Returns a list of available exercises.

        Returns:
            A List with dictionaries containing the exercise information.
        """
        exercises = []

        # Alle Unterverzeichnisse im Übungsverzeichnis durchsuchen
        for exercise_dir in os.listdir(self.exercises_dir):
            config_path = os.path.join(self.exercises_dir, exercise_dir, "config.json")

            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    exercises.append({
                        "id": exercise_dir,
                        "name": config.get("exercise_name", exercise_dir),
                        "description": config.get("description", ""),
                        "config_path": config_path
                    })
                except Exception as e:
                    print(f"Fehler beim Laden der Konfiguration für {exercise_dir}: {e}")

        return exercises

    def get_exercise_config(self, exercise_name):
        """
        Returns the configuration for the specified exercise.

        Args:
            exercise_id: ID of the exercise (directory name)

        Returns:
            Path to the exercise configuration file or a dictionary with the configuration data.
        """
        config = os.path.join(self.exercises_dir, exercise_name, "config.json")

        if isinstance(config, str) and os.path.exists(config):
            with open(config, 'r', encoding='utf-8') as f:
                readyconfig = json.load(f)
        elif isinstance(config, dict):
            readyconfig = config
        else:
            raise ValueError("Konfig muss entweder ein Pfad zu einer JSON-Datei oder ein Dictionary sein")

        return readyconfig

    def create_new_exercise(self, config_data):
        """
        Creates a new exercise with the specified configuration data.

        Args:
            config_data: Dictionary with the exercise configuration data.

        Returns:
            Success status and exercise ID if successful, otherwise error message.
        """
        if "exercise_name" not in config_data:
            return False, "Übungsname fehlt in der Konfiguration"

        exercise_id = config_data.get("id", config_data["exercise_name"].lower().replace(" ", "_"))

        exercise_dir = os.path.join(self.exercises_dir, exercise_id)
        os.makedirs(exercise_dir, exist_ok=True)

        videos_dir = os.path.join(exercise_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        config_path = os.path.join(exercise_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        return True, exercise_id

    def delete_exercise(self, exercise_id):
        """
        Deletes the specified exercise and its contents.

        Args:
            exercise_id: ID of the exercise to delete.

        Returns:
            True -> success, False -> failure
        """
        exercise_dir = os.path.join(self.exercises_dir, exercise_id)

        if not os.path.exists(exercise_dir):
            return False


        for root, dirs, files in os.walk(exercise_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

        os.rmdir(exercise_dir)

        return True

    def get_training_videos_path(self, exercise_id):
        """
        returns the path to the training videos directory for the specified exercise.

        Args:
            exercise_id: ID of the exercise.

        Returns:
            path to the training videos directory.
        """
        return os.path.join(self.exercises_dir, exercise_id, "videos")