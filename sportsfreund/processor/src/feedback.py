"""
Feedback-Generierung für Sportübungen.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .config import EXERCISE_TYPES

class FeedbackGenerator:
    """
    Klasse zur Generierung von Feedback für Sportübungen.
    """

    def __init__(self):
        """Initialisierung des FeedbackGenerators."""
        self.exercise_types = EXERCISE_TYPES
        self.last_feedback_time = 0
        self.feedback_history = []

        # Status für die aktuell laufende Übung
        self.current_exercise_type = None
        self.current_repetition_count = 0
        self.current_state = "pause"  # Mögliche Zustände: "pause", "rep_running", "rep_finished"
        self.previous_state = "pause"

    def set_exercise_type(self, exercise_type: str):
        """
        Setzt den aktuellen Übungstyp.

        Args:
            exercise_type: Typ der Übung (z.B. "squat", "pushup")
        """
        if exercise_type in self.exercise_types:
            self.current_exercise_type = exercise_type
            self.current_repetition_count = 0
            print(f"Übung eingestellt auf: {self.exercise_types[exercise_type]['name']}")
        else:
            raise ValueError(f"Unbekannter Übungstyp: {exercise_type}")

    def process_frame_prediction(self,
                               frame_prediction: int,
                               pose_data: np.ndarray,
                               timestamp: float,
                               delay: float = 1.5) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet die Vorhersage für einen Frame und generiert Feedback, wenn nötig.

        Args:
            frame_prediction: Klassifikation des Frames (0: Pause, 1: Wiederholung läuft, 2: Wiederholung beendet)
            pose_data: Pose-Daten für den aktuellen Frame
            timestamp: Zeitstempel des Frames
            delay: Minimale Verzögerung zwischen Feedback-Nachrichten in Sekunden

        Returns:
            Dictionary mit Feedback-Informationen oder None, wenn kein Feedback generiert wurde
        """
        if self.current_exercise_type is None:
            return None

        # Konvertiere die Vorhersage in einen Zustand
        states = ["pause", "rep_running", "rep_finished"]
        current_frame_state = states[frame_prediction]

        # Aktualisiere den Zustand
        self.previous_state = self.current_state
        self.current_state = current_frame_state

        # Feedback generieren, wenn eine Wiederholung abgeschlossen wurde
        feedback = None
        if self.current_state == "rep_finished" and self.previous_state == "rep_running":
            # Erhöhe den Wiederholungszähler
            self.current_repetition_count += 1

            # Prüfe, ob genug Zeit seit dem letzten Feedback vergangen ist
            if timestamp - self.last_feedback_time >= delay:
                # Analysiere die Ausführungsqualität
                error_type, quality_score = self._analyze_exercise_quality(pose_data)

                # Erstelle das Feedback
                feedback = {
                    "exercise_type": self.current_exercise_type,
                    "exercise_name": self.exercise_types[self.current_exercise_type]['name'],
                    "repetition_count": self.current_repetition_count,
                    "quality_score": quality_score,
                    "error_type": error_type,
                    "timestamp": timestamp
                }

                # Speichere das Feedback in der Historie
                self.feedback_history.append(feedback)

                # Aktualisiere den Zeitstempel für das letzte Feedback
                self.last_feedback_time = timestamp

        return feedback

    def _analyze_exercise_quality(self, pose_data: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Analysiert die Qualität der Übungsausführung basierend auf den Pose-Daten.

        In einer vollständigen Implementierung würde diese Methode spezifische
        Regeln für verschiedene Übungstypen enthalten oder fortgeschrittene ML-Modelle
        zur Fehlererkennung nutzen.

        Args:
            pose_data: Pose-Daten für den aktuellen Frame

        Returns:
            Tuple mit dem erkannten Fehlertyp (oder None, wenn kein Fehler) und
            einem Qualitätswert (0-100)
        """
        if self.current_exercise_type is None:
            return None, 0.0

        # Einfaches Beispiel für die Implementierung für Kniebeugen
        if self.current_exercise_type == "squat":
            return self._analyze_squat_quality(pose_data)
        elif self.current_exercise_type == "pushup":
            return self._analyze_pushup_quality(pose_data)
        else:
            # Standard-Implementierung für andere Übungen
            # In einer vollständigen Anwendung würde hier für jede Übung eine spezifische Analyse stehen
            return None, 85.0

    def _analyze_squat_quality(self, pose_data: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Analysiert die Qualität einer Kniebeuge.

        Args:
            pose_data: Pose-Daten für den aktuellen Frame

        Returns:
            Tuple mit dem erkannten Fehlertyp (oder None, wenn kein Fehler) und
            einem Qualitätswert (0-100)
        """
        # Extrahiere relevante Körperpunkte
        # In einer vollständigen Implementierung würden hier komplexere Berechnungen stehen
        try:
            error_types = self.exercise_types["squat"]["error_types"]

            # Demonstration einer einfachen Regel zur Fehlererkennung
            # In der Realität würden hier fortgeschrittenere biomechanische Analysen stehen

            # Zufällige Auswahl eines Fehlertyps für Demonstrationszwecke
            # In einer realen Anwendung würde die Fehleranalyse auf mathematischer Basis erfolgen
            import random
            if random.random() < 0.7:  # 70% Chance, dass kein Fehler erkannt wird
                return None, random.uniform(80.0, 100.0)
            else:
                error_type = random.choice(error_types)
                quality_score = random.uniform(50.0, 75.0)
                return error_type, quality_score

        except Exception as e:
            print(f"Fehler bei der Analyse der Kniebeuge: {str(e)}")
            return None, 50.0

    def _analyze_pushup_quality(self, pose_data: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Analysiert die Qualität eines Liegestützes.

        Args:
            pose_data: Pose-Daten für den aktuellen Frame

        Returns:
            Tuple mit dem erkannten Fehlertyp (oder None, wenn kein Fehler) und
            einem Qualitätswert (0-100)
        """
        # Ähnlich wie bei der Kniebeuge-Analyse
        try:
            error_types = self.exercise_types["pushup"]["error_types"]

            # Demonstration
            import random
            if random.random() < 0.7:
                return None, random.uniform(80.0, 100.0)
            else:
                error_type = random.choice(error_types)
                quality_score = random.uniform(50.0, 75.0)
                return error_type, quality_score

        except Exception as e:
            print(f"Fehler bei der Analyse des Liegestützes: {str(e)}")
            return None, 50.0

    def get_formatted_feedback(self, feedback: Dict[str, Any]) -> str:
        """
        Formatiert das Feedback für die Anzeige.

        Args:
            feedback: Feedback-Daten

        Returns:
            Formatierte Feedback-Nachricht
        """
        exercise_name = feedback["exercise_name"]
        count = feedback["repetition_count"]
        quality = feedback["quality_score"]
        error_type = feedback["error_type"]

        message = f"Wiederholung {count} von {exercise_name} "

        if error_type:
            message += f"mit Problem: {error_type}. "
            message += f"Qualität: {quality:.1f}%"
        else:
            message += f"korrekt ausgeführt! Qualität: {quality:.1f}%"

        return message

    def get_feedback_history(self) -> List[Dict[str, Any]]:
        """
        Gibt die Historie der generierten Feedbacks zurück.

        Returns:
            Liste der Feedback-Einträge
        """
        return self.feedback_history

    def reset(self):
        """
        Setzt den Feedback-Generator zurück.
        """
        self.current_repetition_count = 0
        self.current_state = "pause"
        self.previous_state = "pause"
        self.feedback_history = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Erstellt eine Zusammenfassung der Übungssitzung.

        Returns:
            Dictionary mit Zusammenfassungsinformationen
        """
        if not self.feedback_history:
            return {
                "exercise_name": "Keine Übung durchgeführt",
                "total_repetitions": 0,
                "average_quality": 0.0,
                "common_errors": []
            }

        # Extrahiere Statistiken aus der Feedback-Historie
        exercise_name = self.feedback_history[0]["exercise_name"]
        total_repetitions = len(self.feedback_history)

        # Berechne die durchschnittliche Qualität
        quality_scores = [fb["quality_score"] for fb in self.feedback_history]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Finde die häufigsten Fehler
        error_types = [fb["error_type"] for fb in self.feedback_history if fb["error_type"] is not None]
        error_counts = {}
        for error in error_types:
            if error in error_counts:
                error_counts[error] += 1
            else:
                error_counts[error] = 1

        # Sortiere Fehler nach Häufigkeit
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "exercise_name": exercise_name,
            "total_repetitions": total_repetitions,
            "average_quality": average_quality,
            "common_errors": common_errors
        }
