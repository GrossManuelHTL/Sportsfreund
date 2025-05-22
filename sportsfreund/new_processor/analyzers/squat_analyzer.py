import logging
import time
import numpy as np
import os
import json
import cv2
from .analyzer_base import AnalyzerBase

class SquatAnalyzer(AnalyzerBase):
    """
    Überarbeitete Klasse zur Analyse von Kniebeugen mit präziserer Phasenerkennung
    und zuverlässiger Fehlererkennung basierend auf config.json.
    """
    def __init__(self, config):
        super().__init__(config)

        # Lade die detaillierten Konfigurationsdaten aus der config.json im squats-Ordner
        self.rep_history = []
        self.config_data = self._load_config()

        # Hilfsvariable für Ausrichtung
        self.lookLeft = False

        # Generiere Feedback-Map aus config_data
        self.feedback_map = self._generate_feedback_map()

        # Historien für die Glättung der Messungen
        self.angle_history = {}
        for angle in self.config_data.get('angles', []):
            self.angle_history[angle] = []

        self.history_size = self.config_data.get('visualization', {}).get('history_size', 5)

        # Tiefste Knieposition für Feedback speichern
        self.lowest_knee_angle = 180

        # Hilfsvariablen für Bewegungsrichtung
        self.last_knee_angles = []

        # Fehlererkennungs-Counter für stabileres Feedback
        self.error_counts = {key: 0 for key in self.feedback_map.keys()}
        self.error_threshold = self.config_data.get('visualization', {}).get('error_persistence_threshold', 3)

        logging.info(f"SquatAnalyzer für {self.config.get_exercise_name()} initialisiert")

        self.last_written_feedback = None
        self.current_feedback = None
        self.feedback_file = "feedback.txt"

    def _generate_feedback_map(self):
        """
        Erstellt die Feedback-Map basierend auf den Daten aus der config.json
        """
        feedback_map = {}
        feedback_rules = self.config_data.get('feedback_rules', {})
        colors = self.config_data.get('visualization', {}).get('colors', {})

        for key, rule in feedback_rules.items():
            description = rule.get('description', '')
            color_name = rule.get('color', 'red')
            priority = rule.get('priority', 100)

            # Konvertiere Farbnamen in RGB-Werte aus der config oder verwende Standardwerte
            if color_name in colors:
                color = tuple(colors[color_name])
            else:
                color = (0, 0, 255)  # Standardfarbe Rot

            feedback_map[key] = (description, priority, color)

        # Füge "good_form" hinzu, falls nicht in der Konfiguration enthalten
        if 'good_form' not in feedback_map:
            feedback_map['good_form'] = ('GUTE AUSFÜHRUNG', 120, tuple(colors.get('green', (0, 255, 0))))

        return feedback_map

    def _write_feedback_to_file(self, feedback_key):
        """
        Schreibt Feedback in eine Datei, aber nur wenn es sich vom letzten Feedback unterscheidet
        """
        if feedback_key == 'good_form' or feedback_key is None:
            return

        if feedback_key != self.last_written_feedback:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            feedback_text = self.feedback_map[feedback_key][0]

            os.makedirs(os.path.dirname(os.path.abspath(self.feedback_file)), exist_ok=True)

            with open(self.feedback_file, "a", encoding="utf-8") as file:
                file.write(f"{timestamp}: {feedback_text}\n")

            self.last_written_feedback = feedback_key
            logging.info(f"Feedback in Datei geschrieben: {feedback_text}")

    def analyze_frame(self, frame, joint_angles, coords, debug=False):
        """
        Analysiert einen Frame und generiert Feedback.
        """
        if joint_angles is None or coords is None:
            return frame, self._get_status()

        # Phase erkennen
        self.previous_phase = self.current_phase
        self.current_phase = self._determine_phase(joint_angles)

        # Phase-Historie aktualisieren
        if self.current_phase:
            if len(self.phase_history) == 0 or self.phase_history[-1] != self.current_phase:
                self.phase_history.append(self.current_phase)
                if self.current_phase in ['standing', 'bottom']:
                    self.rep_history.append(self.current_phase)

                self.state['phase_start_time'] = time.time()
                self.state['phase_count'][self.current_phase] += 1

                if debug:
                    print(f"Phase erkannt: {self.current_phase}")

        # Wiederholungen zählen
        self._count_repetitions(debug)

        # Feedback generieren
        feedback_key = self._generate_feedback(joint_angles, coords)

        # Feedback-Status verfolgen und in Datei schreiben, wenn sich das Feedback ändert
        if feedback_key != self.current_feedback:
            # Feedback hat sich geändert
            if feedback_key != 'good_form' and feedback_key is not None:
                # Neues negatives Feedback - in Datei schreiben
                self._write_feedback_to_file(feedback_key)
            # Aktuelles Feedback aktualisieren
            self.current_feedback = feedback_key

        # Frame mit Feedback und Informationen annotieren
        annotated_frame = self._annotate_frame(frame, feedback_key, joint_angles, debug)

        return annotated_frame, self._get_status()

    def _load_config(self):
        """
        Lädt die Konfigurationsdaten aus der config.json im exercises-Ordner.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'exercises', 'squats', 'config.json')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                logging.info(f"Konfiguration geladen aus: {config_path}")
                return config_data
        except Exception as e:
            logging.error(f"Fehler beim Laden der Konfiguration von {config_path}: {e}")
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
        #logging.info(ratio)
        return ratio < 5

    def _check_knees_over_toes(self, coords, phase):
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


        #logging.info(f"left_knee_x: {left_knee_x}, left_foot_x: {left_foot_x}, right_knee_x: {right_knee_x}, right_foot_x: {right_foot_x}")
        # Knie sollten nicht viel weiter vorne als die Zehen sein

        #logging.info(self.lookLeft)

        if phase == 'standing':
            self.lookLeft = left_foot_x < left_knee_x

        if self.lookLeft:
            ok = left_knee_x >= left_foot_x
        else:
            ok = right_knee_x < right_foot_x

        return ok

    def _check_squat_depth(self, knee_angle):
        #TODO geht nix
        """
        Überprüft, ob die Kniebeuge tief genug ist.
        """
        # Update der tiefsten Position
        logging.info(f"aktuelle Kniebeuge: {knee_angle}, tiefste Position: {self.lowest_knee_angle}")
        self.lowest_knee_angle = min(self.lowest_knee_angle, knee_angle)

        # Tief genug, wenn unter dem Schwellenwert
        deep_enough = knee_angle <= 95
        return deep_enough

    def _determine_phase(self, joint_angles):
        """
        Bestimmt die aktuelle Übungsphase basierend auf den Gelenkwinkeln und den Phasenkriterien aus der Konfiguration.
        """
        if not joint_angles:
            return 'unknown'

        # Gelenkwinkel glätten
        smoothed_angles = self._smooth_angles(joint_angles)

        # Die Bewegungsrichtung basierend auf dem durchschnittlichen Kniewinkel berechnen
        if 'left_knee_angle' in smoothed_angles and 'right_knee_angle' in smoothed_angles:
            avg_knee_angle = (smoothed_angles['left_knee_angle'] + smoothed_angles['right_knee_angle']) / 2
            direction = self._calculate_movement_direction(avg_knee_angle)
        else:
            return 'unknown'

        # Phasenkriterien aus der Konfiguration holen
        phase_criteria = self.config_data.get('phase_criteria', {})

        # Prüfen, welche Phase am besten zu den aktuellen Winkeln passt
        matching_phases = []

        for phase, criteria in phase_criteria.items():
            matches_criteria = True

            # Prüfe alle Winkelkriterien
            for angle_name, angle_range in criteria.items():
                if angle_name == 'direction':
                    # Spezialfall für Richtungskriterium
                    if criteria['direction'] != direction and direction != 'stable':
                        matches_criteria = False
                        break
                elif angle_name in smoothed_angles:
                    angle_value = smoothed_angles[angle_name]
                    min_val, max_val = angle_range

                    if not (min_val <= angle_value <= max_val):
                        matches_criteria = False
                        break

            if matches_criteria:
                matching_phases.append(phase)

        # Wenn mehrere Phasen passen, verwende Kontext oder vorherige Phase zur Entscheidung
        if len(matching_phases) > 1:
            # Wenn 'standing' und 'up' beide passen und die vorherige Phase 'up' war,
            # wähle 'standing' (vollständige Aufwärtsbewegung)
            if 'standing' in matching_phases and 'up' in matching_phases:
                if self.previous_phase == 'up':
                    # Wenn die Aufwärtsbewegung abgeschlossen ist, setze die tiefste Position zurück
                    self.lowest_knee_angle = 180
                    return 'standing'
                else:
                    return 'up'

            # Wenn 'down' und 'up' beide passen, verwende die Richtung
            if 'down' in matching_phases and 'up' in matching_phases:
                if direction == 'down':
                    return 'down'
                elif direction == 'up':
                    return 'up'
                else:
                    # Bei unklarer Richtung, behalte vorherige Phase bei oder wähle 'down' als sicheren Standardwert
                    return self.previous_phase if self.previous_phase in ['down', 'up'] else 'down'

            # Fallback: erste passende Phase
            return matching_phases[0]

        elif len(matching_phases) == 1:
            phase = matching_phases[0]

            # Bei 'standing'-Phase tiefste Position zurücksetzen
            if phase == 'standing' and self.previous_phase == 'up':
                self.lowest_knee_angle = 180

            return phase

        # Wenn keine Phase passt, behalte vorherige Phase bei oder verwende 'unknown'
        return self.previous_phase if self.previous_phase != 'unknown' else 'unknown'

    def _generate_feedback(self, joint_angles, coords):
        """
        Generiert Feedback basierend auf den Regeln aus der Konfiguration.
        """
        if not joint_angles or not coords:
            return None

        # Gelenkwinkel glätten
        smoothed_angles = self._smooth_angles(joint_angles)

        # Feedback-Regeln aus der Konfiguration holen
        feedback_rules = self.config_data.get('feedback_rules', {})

        # Alle Regeln nach Priorität prüfen (niedrigere Zahl = höhere Priorität)
        sorted_rules = sorted(feedback_rules.items(),
                            key=lambda item: item[1].get('priority', 100))

        # Aktuelle Phasen für die Regelauswertung
        current_phase = self.current_phase or 'unknown'

        # Bewegungsmetriken berechnen
        knee_angle = None
        if 'left_knee_angle' in smoothed_angles and 'right_knee_angle' in smoothed_angles:
            knee_angle = (smoothed_angles['left_knee_angle'] + smoothed_angles['right_knee_angle']) / 2

        # Für stabile Erkennung von Feedback-Punkten
        active_feedback = None
        error_detected = False

        for rule_key, rule in sorted_rules:
            # Prüfen, ob die Regel für die aktuelle Phase gilt
            applicable_phases = rule.get('phases', [])
            if applicable_phases and current_phase not in applicable_phases:
                continue

            # Bedingungen der Regel prüfen
            conditions = rule.get('conditions', {})
            condition_met = True

            for condition_key, range_values in conditions.items():
                min_val, max_val = range_values

                # Spezifische Bedingungen überprüfen
                if condition_key == 'shoulder_alignment_ratio':
                    # Prüfe Schulterausrichtung (seitliche Position)
                    if 'left_shoulder' not in coords or 'right_shoulder' not in coords:
                        continue

                    dx = abs(coords['left_shoulder'][0] - coords['right_shoulder'][0])
                    dy = abs(coords['left_shoulder'][1] - coords['right_shoulder'][1])

                    if dy < 10:
                        ratio = 0
                    else:
                        ratio = dx / dy

                    if not (min_val <= ratio <= max_val):
                        condition_met = False

                elif condition_key == 'knee_toe_horizontal_distance':
                    # Prüfe Knie-über-Zehen-Position
                    if not all(k in coords for k in ['left_knee', 'left_foot_index', 'right_knee', 'right_foot_index']):
                        continue

                    left_knee_x = coords['left_knee'][0]
                    left_foot_x = coords['left_foot_index'][0]
                    right_knee_x = coords['right_knee'][0]
                    right_foot_x = coords['right_foot_index'][0]

                    # Knieposition im Verhältnis zu den Zehen prüfen
                    if current_phase == 'standing':
                        self.lookLeft = left_foot_x < left_knee_x

                    if self.lookLeft:
                        distance = left_knee_x - left_foot_x
                    else:
                        distance = right_foot_x - right_knee_x

                    if not (min_val <= distance <= max_val):
                        condition_met = False

                elif condition_key == 'hip_angle':
                    # Prüfe Hüftwinkel (gerader Rücken)
                    hip_angle = None
                    if 'left_hip_angle' in smoothed_angles and 'right_hip_angle' in smoothed_angles:
                        hip_angle = (smoothed_angles['left_hip_angle'] + smoothed_angles['right_hip_angle']) / 2

                    if hip_angle is None:
                        continue

                    if not (min_val <= hip_angle <= max_val):
                        condition_met = False

                elif condition_key == 'lowest_knee_angle':
                    # Prüfe, ob Kniebeuge tief genug war
                    if self.lowest_knee_angle is None:
                        continue

                    if not (min_val <= self.lowest_knee_angle <= max_val):
                        condition_met = False

            # Wenn alle Bedingungen erfüllt sind, Fehler gefunden
            if condition_met:
                # Zähle diesen Fehler
                self.error_counts[rule_key] = self.error_counts.get(rule_key, 0) + 1

                # Zurücksetzen anderer Fehlerzähler
                for other_key in self.error_counts:
                    if other_key != rule_key:
                        self.error_counts[other_key] = 0

                # Prüfe, ob der Fehler lange genug besteht
                if self.error_counts[rule_key] >= self.error_threshold:
                    active_feedback = rule_key
                    error_detected = True
                    break
            else:
                # Fehler tritt nicht auf, reduziere den Zähler
                self.error_counts[rule_key] = max(0, self.error_counts.get(rule_key, 0) - 1)

        # Wenn kein Fehler erkannt wurde, gutes Feedback
        if not error_detected:
            return 'good_form'

        return active_feedback

    def _count_repetitions(self, debug=False):
        """
        Zählt Wiederholungen basierend auf der Phasenhistorie.
        Eine Wiederholung wird nur gezählt, wenn eine vollständige 'standing -> bottom -> standing' Sequenz erkannt wird.
        """
        if len(self.rep_history) >= 3:
            # Suche nach der Sequenz: standing -> bottom -> standing
            last_three = self.rep_history[-3:]

            if last_three == ['standing', 'bottom', 'standing']:
                self.rep_count += 1

                # Behalte nur die letzte Phase (standing) für die nächste Wiederholung
                self.rep_history = [self.rep_history[-1]]

                self.state['last_rep_time'] = time.time()

                if debug:
                    logging.info(f"Wiederholung erkannt! Anzahl: {self.rep_count}")

            # Alternativ: Falls die Historie zu lang wird, kürzen
            elif len(self.rep_history) > 10:
                # Behalte nur die letzten 5 Phasen
                self.rep_history = self.rep_history[-5:]

