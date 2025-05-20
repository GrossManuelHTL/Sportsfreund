import logging
import time
import numpy as np
from analyzer_base import AnalyzerBase

class SquatAnalyzer(AnalyzerBase):
    """
    Überarbeitete Klasse zur Analyse von Kniebeugen mit vereinfachtem Feedback und präziserer Phasenerkennung.
    """
    def __init__(self, config):
        super().__init__(config)

        # Feedback-Map auf die wesentlichen Punkte reduzieren
        self.feedback_map = {
            'knees_too_far': ('KNIE NICHT ÜBER ZEHEN', 170, self.colors['red']),
            'not_deep_enough': ('TIEFER GEHEN', 180, self.colors['yellow']),
            'good_form': ('GUTE AUSFÜHRUNG', 120, self.colors['green'])
        }

        # Historien für die Glättung der Messungen
        self.angle_history = {
            'left_knee_angle': [],
            'right_knee_angle': []
        }
        self.history_size = 5

        # Phasenschwellenwerte
        self.squat_phase_thresholds = {
            'standing': 160,  # Kniewinkel im Stehen
            'bottom': 90      # Kniewinkel im tiefsten Punkt
        }

        logging.info(f"SquatAnalyzer für {self.config.get_exercise_name()} initialisiert")

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

    def detect_exercise_phase(self, joint_angles):
        """
        Präzise Phasenerkennung basierend auf Kniewinkeln.
        """
        if 'left_knee_angle' not in joint_angles or 'right_knee_angle' not in joint_angles:
            return 'unknown'

        knee_angle = (joint_angles['left_knee_angle'] + joint_angles['right_knee_angle']) / 2

        if knee_angle >= self.squat_phase_thresholds['standing']:
            return 'up'
        elif knee_angle <= self.squat_phase_thresholds['bottom']:
            return 'bottom'
        else:
            return 'down'

    def _generate_feedback(self, joint_angles, coords):
        """
        Generiert Feedback basierend auf der Übungsausführung.
        """
        if not joint_angles or not coords:
            return None

        smoothed_angles = self._smooth_angles(joint_angles)

        # Prüfe Knieposition relativ zu den Zehen
        if all(k in coords for k in ['left_knee', 'left_foot_index', 'right_knee', 'right_foot_index']):
            left_knee_x = coords['left_knee'][0]
            left_foot_x = coords['left_foot_index'][0]
            right_knee_x = coords['right_knee'][0]
            right_foot_x = coords['right_foot_index'][0]

            if left_knee_x > left_foot_x + 40 or right_knee_x > right_foot_x + 40:
                return 'knees_too_far'

        # Prüfe Tiefe der Kniebeuge nur, wenn die Phase 'bottom' nicht tief genug war und der Benutzer wieder nach oben geht
        if self.previous_phase == 'bottom' and self.current_phase == 'up':
            if smoothed_angles['left_knee_angle'] > 100 or smoothed_angles['right_knee_angle'] > 100:
                return 'not_deep_enough'

        # Wenn keine Fehler erkannt wurden
        return 'good_form'

