import logging
import time
import numpy as np
import os
import json
from analyzer_base import AnalyzerBase

class SquatAnalyzer(AnalyzerBase):
    """
    Überarbeitete Klasse zur Analyse von Kniebeugen mit präziserer Phasenerkennung
    und zuverlässiger Fehlererkennung für Knieposition und Tiefe.
    """
    def __init__(self, config):
        super().__init__(config)

        # Lade die detaillierten Konfigurationsdaten
        self.config_data = self._load_config()

        # Feedback-Map auf die wesentlichen Punkte reduzieren
        self.feedback_map = {
            'side_position': ('BITTE SEITLICH ZUR KAMERA STEHEN', 190, self.colors['red']),
            'knees_over_toes': ('KNIE NICHT ÜBER DIE ZEHENSPITZEN', 170, self.colors['red']),
            'squat_depth': ('TIEFER GEHEN', 180, self.colors['yellow']),
            'good_form': ('GUTE AUSFÜHRUNG', 120, self.colors['green'])
        }

        # Historien für die Glättung der Messungen
        self.angle_history = {
            'left_knee_angle': [],
            'right_knee_angle': []
        }
        self.history_size = 5

        # Tiefste Knieposition für Feedback speichern
        self.lowest_knee_angle = 180

        # Hilfsvariablen für Bewegungsrichtung
        self.last_knee_angles = []

        logging.info(f"SquatAnalyzer für {self.config.get_exercise_name()} initialisiert")

    def _load_config(self):
        """
        Lädt die Konfigurationsdaten aus der config.json.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'squats', 'config.json')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Fehler beim Laden der Konfiguration: {e}")
            return {}

    def _smooth_angles(self, joint_angles):
        """
        Glättet die Gelenkwinkel durch einen gleitenden Durchschnitt.
        """
        smoothed_angles = {}
        for angle_name, value in joint_angles.items():
            if angle_name in self.angle_history:
                self.angle_history[angle_name].append(value)
                if len(self.angle_history[angle_name]) > self.history_size:
                    self.angle_history[angle_name].pop(0)
                smoothed_angles[angle_name] = np.mean(self.angle_history[angle_name])
            else:
                smoothed_angles[angle_name] = value
        return smoothed_angles

    def _calculate_movement_direction(self, knee_angle):
        """
        Bestimmt die Bewegungsrichtung basierend auf den letzten Kniewinkeln.
        """
        self.last_knee_angles.append(knee_angle)
        if len(self.last_knee_angles) > 5:
            self.last_knee_angles.pop(0)

        if len(self.last_knee_angles) < 3:
            return "unknown"

        # Berechne den Trend der letzten Werte
        recent_diff = self.last_knee_angles[-1] - self.last_knee_angles[-3]

        if recent_diff > 5:
            return "up"
        elif recent_diff < -5:
            return "down"
        else:
            return "stable"

    def _check_side_position(self, coords):
        """
        Überprüft, ob der Benutzer seitlich zur Kamera steht.
        Egal ob links oder rechts ausgerichtet.
        Mit höherer Toleranz für die Ausrichtung.
        """
        if 'left_shoulder' not in coords or 'right_shoulder' not in coords:
            return True

        dx = abs(coords['left_shoulder'][0] - coords['right_shoulder'][0])
        dy = abs(coords['left_shoulder'][1] - coords['right_shoulder'][1])

        if dy < 10:
            ratio = 0
        else:
            ratio = dx / dy

        # tolerance
        return ratio < 1.3

    def _check_knees_over_toes(self, coords):
        """
        Überprüft, ob die Knie über die Zehenspitzen ragen.
        """
        if not all(k in coords for k in ['left_knee', 'left_foot_index', 'right_knee', 'right_foot_index']):
            return True  # Kann nicht überprüfen

        # Berechne den horizontalen Abstand zwischen Knien und Zehen
        left_knee_x = coords['left_knee'][0]
        left_foot_x = coords['left_foot_index'][0]
        right_knee_x = coords['right_knee'][0]
        right_foot_x = coords['right_foot_index'][0]

        # Knie sollten nicht viel weiter vorne als die Zehen sein
        left_ok = left_knee_x <= left_foot_x + 40
        right_ok = right_knee_x <= right_foot_x + 40

        return left_ok and right_ok

    def _check_squat_depth(self, knee_angle):
        """
        Überprüft, ob die Kniebeuge tief genug ist.
        """
        # Update der tiefsten Position
        self.lowest_knee_angle = min(self.lowest_knee_angle, knee_angle)

        # Tief genug, wenn unter dem Schwellenwert
        deep_enough = knee_angle <= 95
        return deep_enough

    def detect_exercise_phase(self, joint_angles):
        """
        Präzise Phasenerkennung basierend auf Kniewinkeln und Bewegungsrichtung.
        """
        if 'left_knee_angle' not in joint_angles or 'right_knee_angle' not in joint_angles:
            return 'unknown'

        knee_angle = (joint_angles['left_knee_angle'] + joint_angles['right_knee_angle']) / 2

        direction = self._calculate_movement_direction(knee_angle)

        # Phase basierend auf Winkeln und Richtung
        if knee_angle >= 165:  # Standing/Up phase
            if self.previous_phase == 'up' or self.previous_phase == 'unknown':
                # Reset tiefste Position bei vollständig aufrechter Position
                self.lowest_knee_angle = 180

            return 'standing'
        elif knee_angle <= 110:  # Bottom phase
            return 'bottom'
        elif direction == 'down':
            return 'down'
        elif direction == 'up':
            return 'up'
        else:
            # Fallback basierend auf vorherigem Zustand und Winkel
            if self.previous_phase == 'down' or self.previous_phase == 'unknown':
                return 'down'
            else:
                return 'up'

    def _generate_feedback(self, joint_angles, coords):
        """
        Generiert Feedback basierend auf der Übungsausführung.
        """
        if not joint_angles or not coords:
            return None

        smoothed_angles = self._smooth_angles(joint_angles)

        # Durchschnittlicher Kniewinkel
        knee_angle = (smoothed_angles['left_knee_angle'] + smoothed_angles['right_knee_angle']) / 2

        # 1. Überprüfe seitliche Position - hat höchste Priorität
        if not self._check_side_position(coords):
            return 'side_position'

        # 2. Überprüfe Knieposition in Down und Bottom Phase
        if self.current_phase in ['down', 'bottom']:
            if not self._check_knees_over_toes(coords):
                return 'knees_over_toes'

        print(self.lowest_knee_angle)

        # 3. Überprüfe Tiefe nach Bottom-Phase beim Hochgehen
        if self.previous_phase == 'bottom' and self.current_phase == 'up':
            if not self._check_squat_depth(self.lowest_knee_angle):
                return 'squat_depth'

        # Keine Probleme gefunden
        return 'good_form'
