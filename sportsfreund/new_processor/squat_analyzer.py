import logging
import time
import numpy as np
from analyzer_base import AnalyzerBase

class SquatAnalyzer(AnalyzerBase):
    """
    Klasse zur Analyse von Übungen auf Basis der erkannten Pose und der Übungskonfiguration.
    Erbt von AnalyzerBase und implementiert übungsspezifische Funktionen.
    """
    def __init__(self, config):
        """
        Initialisiert den ExerciseAnalyzer.

        Args:
            config: ExerciseConfig Objekt
        """
        super().__init__(config)

        # Übungsspezifische Feedback-Map definieren
        self.feedback_map = {
            'back_not_straight': ('RÜCKEN GERADE HALTEN', 215, self.colors['red']),
            'knees_too_far': ('KNIE NICHT ÜBER ZEHEN', 170, self.colors['red']),
            'knees_collapsing': ('KNIE NACH AUSSEN', 160, self.colors['red']),
            'not_deep_enough': ('TIEFER GEHEN', 180, self.colors['yellow']),
            'asymmetrical_squat': ('GLEICHMÄSSIG BELASTEN', 165, self.colors['yellow']),
            'too_fast': ('LANGSAMER AUSFÜHREN', 150, self.colors['yellow']),
            'good_form': ('GUTE AUSFÜHRUNG', 120, self.colors['green'])
        }

        # Historien für die Glättung der Messungen
        self.angle_history = {
            'back_angle': [],
            'left_knee_angle': [],
            'right_knee_angle': [],
            'left_hip_angle': [],
            'right_hip_angle': []
        }
        self.history_size = 5  # Anzahl der Messungen für die Glättung

        # Phasenerkennung verbessern
        self.squat_phase_thresholds = {
            'standing': 160,  # Kniewinkel im Stehen
            'bottom': 90,     # Kniewinkel im tiefsten Punkt
        }

        # Konfidenzgrenzen für zuverlässigere Analyse
        self.min_confidence = 0.6

        logging.info(f"ExerciseAnalyzer für {self.config.get_exercise_name()} initialisiert")

    def _smooth_angles(self, joint_angles):
        """
        Glättet die Gelenkwinkel durch einen gleitenden Durchschnitt

        Args:
            joint_angles: Dictionary der aktuellen Gelenkwinkel

        Returns:
            Dictionary der geglätteten Gelenkwinkel
        """
        smoothed_angles = {}

        for angle_name, value in joint_angles.items():
            if angle_name in self.angle_history:
                # Aktualisiere den Verlauf
                self.angle_history[angle_name].append(value)
                if len(self.angle_history[angle_name]) > self.history_size:
                    self.angle_history[angle_name].pop(0)

                # Berechne den gleitenden Durchschnitt
                smoothed_angles[angle_name] = np.mean(self.angle_history[angle_name])
            else:
                smoothed_angles[angle_name] = value

        return smoothed_angles

    def detect_exercise_phase(self, joint_angles):
        """
        Verbesserte Phasenerkennung für die Kniebeuge.

        Args:
            joint_angles: Dictionary der Gelenkwinkel

        Returns:
            str: Aktuelle Phase ('up', 'down', 'bottom', 'unknown')
        """
        if 'left_knee_angle' not in joint_angles or 'right_knee_angle' not in joint_angles:
            return 'unknown'

        # Durchschnitt beider Kniewinkel verwenden
        knee_angle = (joint_angles['left_knee_angle'] + joint_angles['right_knee_angle']) / 2

        # Phase basierend auf Kniewinkel und Richtung der Bewegung
        if knee_angle >= self.squat_phase_thresholds['standing']:
            return 'up'  # Stehende Position (oben)
        elif knee_angle <= self.squat_phase_thresholds['bottom']:
            return 'bottom'  # Tiefste Position
        else:
            # Bestimme Richtung anhand der bisherigen Phasen
            if self.previous_phase == 'up' or self.previous_phase == 'unknown':
                return 'down'  # Abwärtsbewegung
            else:
                return 'up'  # Aufwärtsbewegung

    def _generate_feedback(self, joint_angles, coords):
        """
        Generiert Feedback basierend auf der Übungsausführung.

        Args:
            joint_angles: Dictionary der Gelenkwinkel
            coords: Dictionary der Landmarken-Koordinaten

        Returns:
            str: Feedback-ID oder None
        """
        if not joint_angles or not coords:
            return None

        # Prüfe Konfidenz der wichtigsten Punkte
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

        confidences = []
        for lm in key_landmarks:
            if lm in coords and len(coords[lm]) > 2:  # Wenn Konfidenzwert vorhanden
                confidences.append(coords[lm][2])

        # Nur analysieren wenn genügend Konfidenz vorhanden
        if confidences and np.mean(confidences) < self.min_confidence:
            return None

        # Glätte die Winkel für stabilere Messungen
        smoothed_angles = self._smooth_angles(joint_angles)

        # Prüfe Regeln für den Rücken
        if 'back_angle' in smoothed_angles:
            back_angle = smoothed_angles['back_angle']

            # Rücken zu stark gekrümmt (nach vorne oder nach hinten)
            if back_angle < 75 or back_angle > 100:
                return 'back_not_straight'

        # Prüfe Regeln für die Knie (für Kniebeugen)
        if 'left_knee_angle' in smoothed_angles and 'right_knee_angle' in smoothed_angles:
            left_knee = smoothed_angles['left_knee_angle']
            right_knee = smoothed_angles['right_knee_angle']

            # Bei zu großem Winkel in der "bottom"-Phase ist die Kniebeuge nicht tief genug
            if self.current_phase == 'bottom' and min(left_knee, right_knee) > 100:
                return 'not_deep_enough'

            # Asymmetrisches Squatten (unterschiedliche Kniewinkel)
            if abs(left_knee - right_knee) > 15:
                return 'asymmetrical_squat'

        # Prüfe die Position der Knie relativ zu den Füßen
        if all(k in coords for k in ['left_knee', 'left_foot_index']) and all(k in coords for k in ['right_knee', 'right_foot_index']):
            left_knee_x, left_knee_y = coords['left_knee'][0:2]
            left_foot_x, left_foot_y = coords['left_foot_index'][0:2]

            right_knee_x, right_knee_y = coords['right_knee'][0:2]
            right_foot_x, right_foot_y = coords['right_foot_index'][0:2]

            # Knie zu weit über den Zehen (für Kniebeugen)
            if (left_knee_x > left_foot_x + 40 or right_knee_x > right_foot_x + 40) and self.current_phase == 'bottom':
                return 'knees_too_far'

            # Prüfung auf einknicken der Knie (nach innen)
            if 'left_ankle' in coords and 'right_ankle' in coords:
                left_ankle_x = coords['left_ankle'][0]
                right_ankle_x = coords['right_ankle'][0]

                # Knie fallen nach innen im Verhältnis zu den Knöcheln
                if ((left_knee_x < left_ankle_x) or (right_knee_x > right_ankle_x)) and self.current_phase == 'bottom':
                    return 'knees_collapsing'

        # Geschwindigkeits-Feedback (basierend auf der Zeit in jeder Phase)
        if self.previous_phase != self.current_phase and self.state['phase_start_time'] > 0:
            phase_duration = time.time() - self.state['phase_start_time']

            # Zu schnelle Bewegung (< 1.2 Sekunde pro Phase)
            if phase_duration < 1.2:
                return 'too_fast'

        # Gute Ausführung, wenn kein negatives Feedback generiert wurde
        return 'good_form'

