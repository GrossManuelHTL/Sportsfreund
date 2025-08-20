"""
Feedback System for Exercise Analysis
Handles real-time error detection and feedback delivery
"""
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import time


class FeedbackType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SAFETY = "safety"


class FeedbackItem:
    """Represents a single feedback item"""
    def __init__(self, message: str, feedback_type: FeedbackType, priority: int = 1):
        self.message = message
        self.type = feedback_type
        self.priority = priority
        self.timestamp = time.time()


class FeedbackHandler:
    """Handles feedback delivery through various channels"""

    def __init__(self):
        self.text_callback: Optional[Callable[[str], None]] = None
        self.audio_callback: Optional[Callable[[str], None]] = None
        self.custom_callback: Optional[Callable[[FeedbackItem], None]] = None

        self.pending_feedback = []
        self.last_safety_feedback = 0
        self.safety_cooldown = 2.0  # seconds

    def set_text_callback(self, callback: Callable[[str], None]):
        """Set callback for text feedback"""
        self.text_callback = callback

    def set_audio_callback(self, callback: Callable[[str], None]):
        """Set callback for audio feedback"""
        self.audio_callback = callback

    def set_custom_callback(self, callback: Callable[[FeedbackItem], None]):
        """Set custom callback for feedback handling"""
        self.custom_callback = callback

    def add_feedback(self, message: str, feedback_type: FeedbackType, immediate: bool = False):
        """Add feedback to the system"""
        feedback_item = FeedbackItem(message, feedback_type)

        if feedback_type == FeedbackType.SAFETY and immediate:
            current_time = time.time()
            if current_time - self.last_safety_feedback > self.safety_cooldown:
                self._deliver_feedback(feedback_item)
                self.last_safety_feedback = current_time
        else:
            self.pending_feedback.append(feedback_item)

    def deliver_pending_feedback(self):
        """Deliver all pending feedback"""
        if not self.pending_feedback:
            return

        # Sort by priority and deliver
        self.pending_feedback.sort(key=lambda x: x.priority, reverse=True)
        for feedback in self.pending_feedback:
            self._deliver_feedback(feedback)

        self.pending_feedback.clear()

    def _deliver_feedback(self, feedback: FeedbackItem):
        """Deliver feedback through registered callbacks"""
        if self.custom_callback:
            self.custom_callback(feedback)
        if self.text_callback:
            self.text_callback(feedback.message)
        if self.audio_callback and feedback.type in [FeedbackType.ERROR, FeedbackType.SAFETY]:
            self.audio_callback(feedback.message)


class ErrorChecker:
    """Checks for form errors based on exercise configuration"""

    def __init__(self, exercise_config: Dict[str, Any]):
        self.exercise_name = exercise_config['name']
        self.error_checks = exercise_config.get('error_checks', [])
        self.feedback_handler = None

        # Collect errors for end of session
        self.collected_errors = []
        self.error_counts = {}

    def set_feedback_handler(self, feedback_handler: FeedbackHandler):
        """Set the feedback handler"""
        self.feedback_handler = feedback_handler

    def check_errors(self, pose_data: Dict[str, Any]) -> List[FeedbackItem]:
        """Check for errors in current pose data - collect only, do NOT show immediately"""
        if not pose_data:
            return []

        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})

        # Collect errors only in background - return NO live errors
        for check in self.error_checks:
            if self._evaluate_condition(check['condition'], angles, positions):
                error_key = check['message']

                # Count errors for statistics
                if error_key not in self.error_counts:
                    self.error_counts[error_key] = 0
                self.error_counts[error_key] += 1

        # ALWAYS return empty list - no live display
        return []

    def get_final_feedback(self) -> List[str]:
        """Get final feedback based on collected errors"""
        feedback_messages = []

        for error_msg, count in self.error_counts.items():
            if count > 20:  # Only mention frequent errors
                if count > 60:
                    feedback_messages.append(f"🔴 Frequent error: {error_msg} ({count}x)")
                elif count > 40:
                    feedback_messages.append(f"🟡 Occasional error: {error_msg} ({count}x)")

        if not feedback_messages:
            feedback_messages.append("✅ Good form! No major issues detected.")

        return feedback_messages

    def reset_errors(self):
        """Reset collected errors"""
        self.collected_errors = []
        self.error_counts = {}

    def _evaluate_condition(self, condition: Dict, angles: Dict, positions: Dict) -> bool:
        """Evaluate a single error condition - less strict"""
        try:
            condition_type = condition.get('type')

            if condition_type == 'angle_range':
                angle_name = condition['angle']
                min_val = condition.get('min', 0)
                max_val = condition.get('max', 180)
                current_angle = angles.get(angle_name, 90)

                # Make bounds less strict (±10° tolerance)
                tolerant_min = min_val - 10
                tolerant_max = max_val + 10

                # Return True if angle is OUTSIDE the tolerant range
                return not (tolerant_min <= current_angle <= tolerant_max)

            elif condition_type == 'position_threshold':
                pos_name = condition['position']
                axis = condition.get('axis', 'y')
                threshold = condition['threshold']
                operator = condition.get('operator', '>')

                current_pos = positions.get(pos_name, {}).get(axis, 0.5)

                # Make thresholds less strict (±0.05 tolerance)
                tolerance = 0.05

                if operator == '>':
                    return current_pos > (threshold + tolerance)
                elif operator == '<':
                    return current_pos < (threshold - tolerance)
                elif operator == '>=':
                    return current_pos >= (threshold + tolerance)
                elif operator == '<=':
                    return current_pos <= (threshold - tolerance)

            return False
        except Exception as e:
            print(f"Error evaluating error condition: {e}")
            return False
