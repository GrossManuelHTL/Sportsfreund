import json
import os


class ExerciseManager:
    """
    Verwaltet alle verfügbaren Übungen im System.
    Ermöglicht das Hinzufügen, Entfernen und Auflisten von Übungen.
    """

    def __init__(self, exercises_dir="exercises"):
        """
        Initialisiert den ExerciseManager.

        Args:
            exercises_dir: Verzeichnis, in dem die Übungskonfigurationen gespeichert sind
        """
        self.exercises_dir = exercises_dir

        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(exercises_dir, exist_ok=True)

    def get_available_exercises(self):
        """
        Gibt eine Liste aller verfügbaren Übungen zurück.

        Returns:
            Liste mit Übungsnamen und Beschreibungen
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
        Gibt die Konfiguration für eine bestimmte Übung zurück.

        Args:
            exercise_id: ID der Übung (Verzeichnisname)

        Returns:
            Pfad zur Konfigurationsdatei oder None, wenn nicht gefunden
            :param exercise_name:
        """
        config = os.path.join(self.exercises_dir, exercise_name, "config.json")
        print(config)

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
        Erstellt eine neue Übung basierend auf Konfigurationsdaten.

        Args:
            config_data: Dictionary mit Konfigurationsdaten

        Returns:
            Erfolgs-Status und Fehlermeldung (falls vorhanden)
        """
        if "exercise_name" not in config_data:
            return False, "Übungsname fehlt in der Konfiguration"

        # ID aus dem Namen erstellen (Kleinbuchstaben, Leerzeichen durch Unterstriche ersetzen)
        exercise_id = config_data.get("id", config_data["exercise_name"].lower().replace(" ", "_"))

        # Verzeichnis erstellen
        exercise_dir = os.path.join(self.exercises_dir, exercise_id)
        os.makedirs(exercise_dir, exist_ok=True)

        # Videos-Verzeichnis erstellen
        videos_dir = os.path.join(exercise_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        # Konfiguration speichern
        config_path = os.path.join(exercise_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        return True, exercise_id

    def delete_exercise(self, exercise_id):
        """
        Löscht eine Übung und alle zugehörigen Dateien.

        Args:
            exercise_id: ID der Übung

        Returns:
            True bei Erfolg, False sonst
        """
        exercise_dir = os.path.join(self.exercises_dir, exercise_id)

        if not os.path.exists(exercise_dir):
            return False

        # Alle Dateien im Verzeichnis rekursiv löschen
        for root, dirs, files in os.walk(exercise_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

        # Verzeichnis selbst löschen
        os.rmdir(exercise_dir)

        return True

    def get_training_videos_path(self, exercise_id):
        """
        Gibt den Pfad zum Verzeichnis mit den Trainingsvideos zurück.

        Args:
            exercise_id: ID der Übung

        Returns:
            Pfad zum Videos-Verzeichnis
        """
        return os.path.join(self.exercises_dir, exercise_id, "videos")