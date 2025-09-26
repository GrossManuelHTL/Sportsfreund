# python
# Datei: `pipeline/video_source.py`
import cv2
from typing import Optional


class VideoSource:
    def __init__(self, path: Optional[str] = None, camera_index: int = 0):
        self.path = path
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        if self.path:
            self.cap = cv2.VideoCapture(self.path)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_fps(self):
        if not self.cap:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        if not self.cap:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
