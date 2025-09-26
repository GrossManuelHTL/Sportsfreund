from enum import Enum
import time

class FeedbackType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SAFETY = "safety"


class FeedbackItem:
    """Represents a single feedback item."""

    def __init__(self, message: str, feedback_type: FeedbackType, priority: int = 1):
        self.message = message
        self.type = feedback_type
        self.priority = priority
        self.timestamp = time.time()
