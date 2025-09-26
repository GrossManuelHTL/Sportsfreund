# python
# Datei: `pipeline/visualizer.py`
import cv2
from typing import Optional, Dict, Any


class Visualizer:
    WINDOW_NAME = "Exercise Analysis"

    def __init__(self, window_name: Optional[str] = None):
        if window_name:
            self.WINDOW_NAME = window_name
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def render(self, frame, pose_data: Optional[Dict] = None, status: Optional[Dict] = None,
               errors: Optional[list] = None, paused: bool = False):
        h, w = frame.shape[:2]

        # simple status panel (keine komplette große Funktion kopiert)
        if status:
            state = status.get('state', 'n/a')
            reps = status.get('reps', 0)
            frame_idx = status.get('frame', 0)
            cv2.putText(frame, f"{state} | Reps: {reps} | F:{frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        cv2.imshow(self.WINDOW_NAME, frame)

    def handle_keys(self):
        """Return key code or None. Caller decides actions (pause/quit)."""
        key = cv2.waitKey(1) & 0xFF
        return key

    def destroy(self):
        cv2.destroyWindow(self.WINDOW_NAME)
