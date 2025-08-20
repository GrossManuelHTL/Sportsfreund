"""
Session Management System
Manages training sessions with sets, reps and feedback collection
"""
import json
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class SetData:
    """Data for a single set"""
    set_number: int
    target_reps: int
    completed_reps: int
    duration_seconds: float
    feedback_items: List[Dict[str, Any]]
    form_score: float  # 0.0 - 1.0
    start_time: datetime
    end_time: Optional[datetime] = None


@dataclass
class SessionData:
    """Data for a complete training session"""
    session_id: str
    exercise_name: str
    target_sets: int
    target_reps_per_set: int
    sets: List[SetData]
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    overall_form_score: float = 0.0


class SessionManager:
    """Manages training sessions and collects feedback"""

    def __init__(self, audio_system=None):
        self.audio_system = audio_system
        self.current_session: Optional[SessionData] = None
        self.current_set: Optional[SetData] = None
        self.session_feedback: List[Dict[str, Any]] = []

    def start_session(self, exercise_name: str, target_sets: int, target_reps_per_set: int) -> str:
        """
        Starts a new training session

        Args:
            exercise_name: Name of the exercise
            target_sets: Number of planned sets
            target_reps_per_set: Number of repetitions per set

        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())

        self.current_session = SessionData(
            session_id=session_id,
            exercise_name=exercise_name,
            target_sets=target_sets,
            target_reps_per_set=target_reps_per_set,
            sets=[],
            start_time=datetime.now()
        )

        print(f"🎯 Session started: {exercise_name} - {target_sets} sets of {target_reps_per_set} reps")

        if self.audio_system:
            self.audio_system.speak(
                f"Trainingseinheit gestartet. {exercise_name}. "
                f"{target_sets} Sets mit jeweils {target_reps_per_set} Wiederholungen."
            )

        return session_id

    def start_set(self, set_number: int) -> bool:
        """
        Starts a new set

        Args:
            set_number: Set number (1-based)

        Returns:
            bool: True if successfully started
        """
        if not self.current_session:
            print("❌ No active session")
            return False

        self.current_set = SetData(
            set_number=set_number,
            target_reps=self.current_session.target_reps_per_set,
            completed_reps=0,
            duration_seconds=0.0,
            feedback_items=[],
            form_score=0.0,
            start_time=datetime.now()
        )

        print(f"🏋️ Set {set_number} started")

        if self.audio_system:
            self.audio_system.speak(f"Set {set_number} beginnt jetzt. Bereit?")

        return True

    def add_feedback(self, feedback_type: str, message: str, severity: str = "info"):
        """
        Adds feedback to the current set

        Args:
            feedback_type: Type of feedback (form, timing, safety, etc.)
            message: Feedback message
            severity: Severity (info, warning, error)
        """
        if not self.current_set:
            return

        feedback_item = {
            "timestamp": time.time(),
            "type": feedback_type,
            "message": message,
            "severity": severity
        }

        self.current_set.feedback_items.append(feedback_item)
        self.session_feedback.append(feedback_item)

    def update_rep_count(self, completed_reps: int):
        """Updates the number of completed repetitions"""
        if self.current_set:
            self.current_set.completed_reps = completed_reps

    def update_form_score(self, score: float):
        """Updates the form score (0.0 - 1.0)"""
        if self.current_set:
            self.current_set.form_score = score

    def finish_set(self) -> Dict[str, Any]:
        """
        Finishes the current set and returns feedback

        Returns:
            Dict: Set summary and feedback
        """
        if not self.current_set or not self.current_session:
            return {}

        # Finish set
        self.current_set.end_time = datetime.now()
        self.current_set.duration_seconds = (
            self.current_set.end_time - self.current_set.start_time
        ).total_seconds()

        # Add set to session
        self.current_session.sets.append(self.current_set)

        # Compile feedback
        set_summary = {
            "set_number": self.current_set.set_number,
            "completed_reps": self.current_set.completed_reps,
            "target_reps": self.current_set.target_reps,
            "duration": self.current_set.duration_seconds,
            "form_score": self.current_set.form_score,
            "feedback_count": len(self.current_set.feedback_items)
        }

        print(f"✅ Set {self.current_set.set_number} completed: "
              f"{self.current_set.completed_reps}/{self.current_set.target_reps} reps, "
              f"Form: {self.current_set.form_score:.1%}")

        # Audio feedback
        if self.audio_system:
            self._speak_set_feedback(set_summary)

        # Reset current set
        current_set_data = self.current_set
        self.current_set = None

        return set_summary

    def _speak_set_feedback(self, set_summary: Dict[str, Any]):
        """Speaks the set feedback"""
        if not self.audio_system:
            return

        reps_completed = set_summary["completed_reps"]
        target_reps = set_summary["target_reps"]
        form_score = set_summary["form_score"]

        # Basic feedback
        feedback_text = f"Set {set_summary['set_number']} abgeschlossen. "
        feedback_text += f"{reps_completed} von {target_reps} Wiederholungen. "

        # Form evaluation
        if form_score >= 0.8:
            feedback_text += "Ausgezeichnete Ausführung! "
        elif form_score >= 0.6:
            feedback_text += "Gute Ausführung. "
        elif form_score >= 0.4:
            feedback_text += "Ausführung kann verbessert werden. "
        else:
            feedback_text += "Achte mehr auf die korrekte Ausführung. "

        # Specific feedback from collected items
        feedback_items = self.current_session.sets[-1].feedback_items if self.current_session.sets else []
        warning_count = len([f for f in feedback_items if f["severity"] == "warning"])
        error_count = len([f for f in feedback_items if f["severity"] == "error"])

        if error_count > 0:
            feedback_text += f"{error_count} Ausführungsfehler erkannt. "
        elif warning_count > 0:
            feedback_text += f"{warning_count} Verbesserungshinweise. "

        self.audio_system.speak(feedback_text)

    def finish_session(self) -> SessionData:
        """
        Finishes the current session

        Returns:
            SessionData: Complete session data
        """
        if not self.current_session:
            return None

        self.current_session.end_time = datetime.now()
        self.current_session.total_duration_seconds = (
            self.current_session.end_time - self.current_session.start_time
        ).total_seconds()

        # Calculate overall score
        if self.current_session.sets:
            total_score = sum(s.form_score for s in self.current_session.sets)
            self.current_session.overall_form_score = total_score / len(self.current_session.sets)

        print(f"🏁 Session completed: {len(self.current_session.sets)} sets, "
              f"Overall score: {self.current_session.overall_form_score:.1%}")

        # Final audio feedback
        if self.audio_system:
            self._speak_session_summary()

        session_data = self.current_session
        self.current_session = None
        self.session_feedback = []

        return session_data

    def _speak_session_summary(self):
        """Speaks the session summary"""
        if not self.audio_system or not self.current_session:
            return

        total_sets = len(self.current_session.sets)
        total_reps = sum(s.completed_reps for s in self.current_session.sets)
        avg_score = self.current_session.overall_form_score
        duration_minutes = self.current_session.total_duration_seconds / 60

        summary_text = f"Training abgeschlossen! "
        summary_text += f"{total_sets} Sets mit insgesamt {total_reps} Wiederholungen. "
        summary_text += f"Trainingsdauer: {duration_minutes:.1f} Minuten. "

        if avg_score >= 0.8:
            summary_text += "Hervorragende Leistung! "
        elif avg_score >= 0.6:
            summary_text += "Gute Leistung! "
        else:
            summary_text += "Weiter so, beim nächsten Mal wird es noch besser! "

        self.audio_system.speak(summary_text)

    def get_session_data_for_backend(self) -> Dict[str, Any]:
        """
        Prepares session data for backend upload

        Returns:
            Dict: Serialized session data
        """
        if not self.current_session:
            return {}

        return {
            "session": asdict(self.current_session),
            "feedback_summary": {
                "total_feedback_items": len(self.session_feedback),
                "warning_count": len([f for f in self.session_feedback if f["severity"] == "warning"]),
                "error_count": len([f for f in self.session_feedback if f["severity"] == "error"]),
            }
        }

    def is_session_active(self) -> bool:
        """Checks if a session is active"""
        return self.current_session is not None

    def is_set_active(self) -> bool:
        """Checks if a set is active"""
        return self.current_set is not None
