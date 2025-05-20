import logging
import time
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
            'not_deep_enough': ('TIEFER GEHEN', 180, self.colors['yellow']),
            'too_fast': ('LANGSAMER AUSFÜHREN', 150, self.colors['yellow']),
            'good_form': ('GUTE AUSFÜHRUNG', 120, self.colors['green'])
        }

        logging.info(f"ExerciseAnalyzer für {self.config.get_exercise_name()} initialisiert")

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

        # Prüfe Regeln für den Rücken
        if 'back_angle' in joint_angles:
            back_angle = joint_angles['back_angle']

            # Rücken zu stark gekrümmt (nach vorne oder nach hinten)
            if back_angle < 80 or back_angle > 100:
                return 'back_not_straight'

        # Prüfe Regeln für die Knie (für Kniebeugen)
        if 'left_knee_angle' in joint_angles and 'right_knee_angle' in joint_angles:
            left_knee = joint_angles['left_knee_angle']
            right_knee = joint_angles['right_knee_angle']

            # Bei zu großem Winkel in der "bottom"-Phase ist die Kniebeuge nicht tief genug
            if self.current_phase == 'bottom' and (left_knee > 100 or right_knee > 100):
                return 'not_deep_enough'

        # Prüfe die Position der Knie relativ zu den Füßen
        if all(k in coords for k in ['left_knee', 'left_foot_index']) and all(k in coords for k in ['right_knee', 'right_foot_index']):
            left_knee_x = coords['left_knee'][0]
            left_foot_x = coords['left_foot_index'][0]

            right_knee_x = coords['right_knee'][0]
            right_foot_x = coords['right_foot_index'][0]

            # Knie zu weit über den Zehen (für Kniebeugen)
            if left_knee_x > left_foot_x + 50 or right_knee_x > right_foot_x + 50:
                return 'knees_too_far'

        # Geschwindigkeits-Feedback (basierend auf der Zeit in jeder Phase)
        if self.previous_phase != self.current_phase and self.state['phase_start_time'] > 0:
            phase_duration = time.time() - self.state['phase_start_time']

            # Zu schnelle Bewegung (< 1 Sekunde pro Phase)
            if phase_duration < 1.0:
                return 'too_fast'

        # Gute Ausführung, wenn kein negatives Feedback generiert wurde
        return 'good_form'

