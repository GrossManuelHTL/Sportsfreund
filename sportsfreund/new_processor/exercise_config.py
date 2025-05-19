import os
import json
import logging

class ExerciseConfig:
    """
    Klasse zum Laden und Verwalten von Übungskonfigurationen.
    """
    def __init__(self, config_path=None):
        """
        Initialisiert die Konfigurationsklasse.

        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """
        Lädt die Konfiguration aus einer JSON-Datei.

        Args:
            config_path: Pfad zur Konfigurationsdatei

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            if not os.path.exists(config_path):
                logging.error(f"Konfigurationsdatei nicht gefunden: {config_path}")
                return False

            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            # Konfiguration validieren
            if not self._validate_config():
                logging.error(f"Ungültige Konfiguration: {config_path}")
                return False

            logging.info(f"Konfiguration erfolgreich geladen: {config_path}")
            return True

        except Exception as e:
            logging.error(f"Fehler beim Laden der Konfiguration: {e}")
            return False

    def _validate_config(self):
        """
        Validiert die Konfigurationsdaten.

        Returns:
            bool: True wenn gültig, False sonst
        """
        required_fields = ['name', 'joints', 'phases', 'phase_criteria']

        if not self.config:
            return False

        # Prüfe, ob alle erforderlichen Felder vorhanden sind
        for field in required_fields:
            if field not in self.config:
                logging.error(f"Fehlendes Feld in Konfiguration: {field}")
                return False

        # Prüfe, ob alle Phasen in phase_criteria vorhanden sind
        for phase in self.config['phases']:
            if phase not in self.config['phase_criteria']:
                logging.error(f"Phase {phase} nicht in phase_criteria definiert")
                return False

        return True

    def get_config(self):
        """
        Gibt die aktuelle Konfiguration zurück.

        Returns:
            dict: Die Konfiguration
        """
        return self.config

    def get_exercise_name(self):
        """
        Gibt den Namen der Übung zurück.

        Returns:
            str: Name der Übung
        """
        if not self.config:
            return None
        return self.config.get('name')

    def get_joints(self):
        """
        Gibt die relevanten Gelenke für die Übung zurück.

        Returns:
            list: Liste der relevanten Gelenke
        """
        if not self.config:
            return []
        return self.config.get('joints', [])

    def get_phases(self):
        """
        Gibt die definierten Phasen der Übung zurück.

        Returns:
            list: Liste der Phasen
        """
        if not self.config:
            return []
        return self.config.get('phases', [])

    def get_phase_criteria(self, phase=None):
        """
        Gibt die Kriterien für eine bestimmte Phase zurück.

        Args:
            phase: Name der Phase oder None für alle Phasen

        Returns:
            dict: Kriterien für die Phase(n)
        """
        if not self.config:
            return {}

        criteria = self.config.get('phase_criteria', {})

        if phase:
            return criteria.get(phase, {})
        return criteria

    def get_tolerance(self):
        """
        Gibt den Toleranzwert für die Übung zurück.

        Returns:
            float: Toleranzwert
        """
        if not self.config:
            return 0.1
        return self.config.get('tolerance', 0.1)

    def get_feedback_rules(self):
        """
        Gibt die Feedback-Regeln für die Übung zurück.

        Returns:
            dict: Feedback-Regeln
        """
        if not self.config:
            return {}
        return self.config.get('feedback_rules', {})
