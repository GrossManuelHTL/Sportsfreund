from typing import Dict, List, Any, Callable, Optional
from ...models.feedback import FeedbackItem, FeedbackType
from collections import deque, defaultdict
import time

class FeedbackHandler:
    """Handles feedback delivery through various channels."""

    def __init__(self):
        self.text_callback: Optional[Callable[[str], None]] = None
        self.audio_callback: Optional[Callable[[str], None]] = None
        self.custom_callback: Optional[Callable[[FeedbackItem], None]] = None

        self.pending_feedback: List[FeedbackItem] = []
        self.last_safety_feedback = 0
        self.safety_cooldown = 2.0  # seconds

        # recent display cache (message, type, timestamp)
        self.recent_display: deque = deque(maxlen=10)
        self.display_ttl = 3.0  # seconds

    def set_text_callback(self, callback: Callable[[str], None]):
        """Set callback for text feedback."""
        self.text_callback = callback

    def set_audio_callback(self, callback: Callable[[str], None]):
        """Set callback for audio feedback."""
        self.audio_callback = callback

    def set_custom_callback(self, callback: Callable[[FeedbackItem], None]):
        """Set custom callback for feedback handling."""
        self.custom_callback = callback

    def add_feedback(self, message: str, feedback_type: FeedbackType, immediate: bool = False, priority: int = 1):
        """Add feedback to the system."""
        feedback_item = FeedbackItem(message, feedback_type, priority)
        # If caller requests immediate delivery, deliver now regardless of type
        if immediate:
            try:
                self._deliver_feedback(feedback_item)
            except Exception:
                pass
            # update safety timestamp when applicable
            if feedback_type == FeedbackType.SAFETY:
                self.last_safety_feedback = time.time()
            return

        # default: safety gets a cooldown-protected immediate delivery if requested by callers elsewhere
        if feedback_type == FeedbackType.SAFETY:
            current_time = time.time()
            if current_time - self.last_safety_feedback > self.safety_cooldown:
                self._deliver_feedback(feedback_item)
                self.last_safety_feedback = current_time
                return

        # otherwise queue for later delivery
        self.pending_feedback.append(feedback_item)

    def deliver_pending_feedback(self):
        """Deliver all pending feedback."""
        if not self.pending_feedback:
            return

        # Sort by priority (higher first) and deliver
        self.pending_feedback.sort(key=lambda x: x.priority, reverse=True)
        for feedback in self.pending_feedback:
            self._deliver_feedback(feedback)

        self.pending_feedback.clear()

    def _deliver_feedback(self, feedback: FeedbackItem):
        """Deliver feedback through registered callbacks."""
        # assign to locals and call if present to avoid static analysis false-positives
        cb_custom = self.custom_callback
        cb_text = self.text_callback

        if cb_custom:
            try:
                cb_custom(feedback)
            except Exception:
                pass

        if cb_text:
            try:
                cb_text(feedback.message)
            except Exception:
                pass

        # store for on-screen display with timestamp
        try:
            self.recent_display.append({'message': feedback.message, 'type': feedback.type.value, 'timestamp': time.time()})
        except Exception:
            pass