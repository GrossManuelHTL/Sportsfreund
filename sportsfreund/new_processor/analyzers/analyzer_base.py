import numpy as np
import logging
import time
import cv2
from abc import ABC, abstractmethod

class AnalyzerBase(ABC):
    """
    Abstrakte Basisklasse für Übungsanalyzer.
    Definiert die grundlegende Struktur und Funktionalität für alle Übungsanalysen.
    """
    def __init__(self, config):
        """
        Initialisiert den Basis-Analyzer.

        Args:
            config: ExerciseConfig Objekt
        """
        self.config = config
        self.current_phase = None
        self.previous_phase = None
        self.rep_count = 0
        self.incorrect_rep_count = 0
        self.phase_history = []
        self.feedback_history = []

        # Standardfarben für die visuelle Ausgabe
        self.colors = {
            'blue': (255, 127, 0),  # In OpenCV ist die Farbreihenfolge BGR
            'red': (50, 50, 255),
            'green': (127, 255, 0),
            'light_green': (127, 233, 100),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (255, 255, 0),
            'light_blue': (255, 204, 102)
        }

        # Zustandsvariablen
        self.state = {
            'start_time': time.time(),
            'last_rep_time': time.time(),
            'phase_start_time': time.time(),
            'in_position': False,
            'display_feedback': [],
            'feedback_times': {},  # Zeitpunkt des letzten Feedbacks für jede Feedback-ID
            'feedback_duration': 3.0,  # Feedback wird 3 Sekunden lang angezeigt
            'phase_count': {phase: 0 for phase in self.config.get_phases()}
        }

        # Feedback-Definitionskarte - kann in abgeleiteten Klassen überschrieben werden
        self.feedback_map = {}

        logging.info(f"AnalyzerBase für {self.config.get_exercise_name()} initialisiert")

    def analyze_frame(self, frame, joint_angles, coords, debug=True):
        """
        Analysiert einen Frame basierend auf den berechneten Gelenkwinkeln.

        Args:
            frame: Der aktuelle Videoframe
            joint_angles: Dictionary der berechneten Gelenkwinkel
            coords: Dictionary der Landmarken-Koordinaten
            debug: Debug-Ausgaben aktivieren

        Returns:
            tuple: (Annotierter Frame, Status-Dictionary)
        """
        if joint_angles is None or coords is None:
            return frame, self._get_status()

        # Bestimme die aktuelle Phase basierend auf den Gelenkwinkeln
        self.previous_phase = self.current_phase
        self.current_phase = self._determine_phase(joint_angles)

        # Aktualisiere Phase-Historie
        if self.current_phase:
            if len(self.phase_history) == 0 or self.phase_history[-1] != self.current_phase:
                self.phase_history.append(self.current_phase)
                self.state['phase_start_time'] = time.time()
                self.state['phase_count'][self.current_phase] += 1

                if debug:
                    print(f"Phase erkannt: {self.current_phase}")

        # Zähle Wiederholungen
        self._count_repetitions(debug)

        # Generiere Feedback basierend auf der Ausführung
        feedback_id = self._generate_feedback(joint_angles, coords)
        #logging.info(feedback_id)

        # Annotiere Frame mit Informationen
        annotated_frame = self._annotate_frame(frame, feedback_id, joint_angles, debug)

        return annotated_frame, self._get_status()

    def _determine_phase(self, joint_angles):
        """
        Bestimmt die aktuelle Phase der Übung basierend auf den Gelenkwinkeln.

        Args:
            joint_angles: Dictionary der Gelenkwinkel

        Returns:
            str: Name der aktuellen Phase oder None
        """
        if not joint_angles:
            return None

        # Hole Phasenkriterien aus der Konfiguration
        phase_criteria = self.config.get_phase_criteria()
        tolerance = self.config.get_tolerance()

        for phase, criteria in phase_criteria.items():
            phase_match = True

            for angle_name, angle_range in criteria.items():
                if angle_name in joint_angles:
                    angle = joint_angles[angle_name]
                    min_val, max_val = angle_range

                    # Prüfe, ob der Winkel im erlaubten Bereich liegt (mit Toleranz)
                    if not (min_val - tolerance * min_val <= angle <= max_val + tolerance * max_val):
                        phase_match = False
                        break
                else:
                    # Winkel nicht verfügbar, Phase kann nicht bestimmt werden
                    phase_match = False
                    break

            if phase_match:
                return phase

        return None

    def _count_repetitions(self, debug=False):
        """
        Zählt Wiederholungen basierend auf der Phasenhistorie.
        Die Basisimplementierung erwartet eine Sequenz von Phasen, die in der Konfiguration definiert ist.

        Args:
            debug: Debug-Ausgaben aktivieren
        """
        # Prüfe nach der Wiederholungssequenz in der Konfiguration
        sequence = self.config.get_config().get('repetition_sequence', [])

        # Wenn keine Sequenz definiert ist, verwende einen Standardalgorithmus
        if not sequence:
            phases = self.config.get_phases()

            if len(phases) >= 2 and len(self.phase_history) >= 3:
                # Einfache Erkennung: [Startphase -> Mittlere Phase(n) -> Startphase]
                start_phase = phases[0]

                # Prüfe, ob wir eine Start-Mittel-Start Sequenz haben
                if (self.phase_history[-3] == start_phase and
                    self.phase_history[-2] != start_phase and
                    self.phase_history[-1] == start_phase):

                    self.rep_count += 1
                    # Entferne die erkannte Sequenz aus der Historie, behalte aber die letzte Phase bei
                    self.phase_history = self.phase_history[-1:]

                    self.state['last_rep_time'] = time.time()

                    if debug:
                        print(f"Wiederholung erkannt (Standard)! Anzahl: {self.rep_count}")

            return

        # Ansonsten verwende die definierte Sequenz
        if len(self.phase_history) < len(sequence):
            return

        # Prüfe, ob der letzte Teil der Historie mit der Sequenz übereinstimmt
        last_phases = self.phase_history[-(len(sequence)):]

        if last_phases == sequence:
            self.rep_count += 1
            # Entferne alle Phasen außer der letzten (die zum nächsten Rep gehören könnte)
            self.phase_history = [self.phase_history[-1]]
            self.state['last_rep_time'] = time.time()

            if debug:
                print(f"Wiederholung erkannt (Sequenz)! Anzahl: {self.rep_count}")

    @abstractmethod
    def _generate_feedback(self, joint_angles, coords):
        """
        Generiert Feedback basierend auf der Übungsausführung.
        Diese Methode sollte in abgeleiteten Klassen überschrieben werden.

        Args:
            joint_angles: Dictionary der Gelenkwinkel
            coords: Dictionary der Landmarken-Koordinaten

        Returns:
            str: Feedback-ID oder None
        """
        pass

    def _annotate_frame(self, frame, feedback_ids, joint_angles, debug=True):
        """
        Annotiert den Frame mit Informationen zur Übung.

        Args:
            frame: Der aktuelle Videoframe
            feedback_ids: Die ID der aktuellen Feedbacks
            joint_angles: Dictionary der Gelenkwinkel
            debug: Debug-Infos anzeigen

        Returns:
            numpy.ndarray: Annotierter Frame
        """
        # Kopie des Frames erstellen
        annotated = frame.copy()

        # Bildgröße ermitteln
        h, w, _ = annotated.shape

        # Wiederholungszähler anzeigen
        cv2.putText(
            annotated,
            f"Reps: {self.rep_count}",
            (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.colors['white'],
            2,
            cv2.LINE_AA
        )

        # Aktuelle Phase anzeigen
        if self.current_phase:
            cv2.putText(
                annotated,
                f"Phase: {self.current_phase}",
                (w - 200, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.colors['yellow'],
                2,
                cv2.LINE_AA
            )

        # Feedback anzeigen
        for feedback_id in feedback_ids:
            if feedback_id and feedback_id in self.feedback_map:
                text, y_pos, color = self.feedback_map[feedback_id]

                # Aktuelle Zeit
                current_time = time.time()

                # Neue Feedback-Nachricht hinzufügen
                if feedback_id not in self.state['feedback_times'] or \
                   current_time - self.state['feedback_times'].get(feedback_id, 0) > self.state['feedback_duration']:
                    self.state['feedback_times'][feedback_id] = current_time

                    # Feedback zur Historie hinzufügen, wenn es sich geändert hat
                    if len(self.feedback_history) == 0 or self.feedback_history[-1] != feedback_id:
                        self.feedback_history.append(feedback_id)

                # Alle aktiven Feedback-Nachrichten anzeigen
                active_feedbacks = []
                for fb_id, timestamp in self.state['feedback_times'].items():
                    if current_time - timestamp <= self.state['feedback_duration']:
                        active_feedbacks.append(fb_id)

                # Anzeigen der aktiven Feedbacks
                for i, fb_id in enumerate(active_feedbacks):
                    fb_text, fb_y_pos, fb_color = self.feedback_map[fb_id]
                    y_position = fb_y_pos + i * 40  # Versetzt die Textzeilen

                    cv2.putText(
                        annotated,
                        fb_text,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        fb_color,
                        2,
                        cv2.LINE_AA
                    )

        # Debug-Informationen anzeigen
        if debug and joint_angles:
            debug_y = 120
            for angle_name, angle_value in joint_angles.items():
                cv2.putText(
                    annotated,
                    f"{angle_name}: {angle_value:.1f}°",
                    (10, debug_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.colors['white'],
                    1,
                    cv2.LINE_AA
                )
                debug_y += 25

        return annotated

    def _get_status(self):
        """
        Gibt ein Status-Dictionary mit aktuellen Informationen zur Übungsausführung zurück.

        Returns:
            dict: Status-Dictionary
        """
        return {
            'exercise': self.config.get_exercise_name(),
            'rep_count': self.rep_count,
            'incorrect_rep_count': self.incorrect_rep_count,
            'current_phase': self.current_phase,
            'phase_history': self.phase_history.copy(),
            'feedback_history': self.feedback_history.copy(),
            'time_elapsed': time.time() - self.state['start_time'],
            'time_since_last_rep': time.time() - self.state['last_rep_time'],
        }

    def reset(self):
        """
        Setzt den Analysezustand zurück.
        """
        self.current_phase = None
        self.previous_phase = None
        self.rep_count = 0
        self.incorrect_rep_count = 0
        self.phase_history = []
        self.feedback_history = []

        self.state = {
            'start_time': time.time(),
            'last_rep_time': time.time(),
            'phase_start_time': time.time(),
            'in_position': False,
            'display_feedback': [],
            'feedback_times': {},
            'feedback_duration': 3.0,
            'phase_count': {phase: 0 for phase in self.config.get_phases()}
        }

        logging.info("Analyzer zurückgesetzt")
