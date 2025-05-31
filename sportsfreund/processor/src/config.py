"""
Konfigurationsdatei für den AI-Coach.
"""

import os

# Pfad-Konfigurationen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(BASE_DIR, "testdata")

# Stellen Sie sicher, dass die Verzeichnisse existieren
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Video-Konfiguration
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30

# MediaPipe-Konfiguration
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0, 1, oder 2
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Modellkonfiguration
SEQUENCE_LENGTH = 30  # Anzahl der Frames für eine Sequenz
NUM_KEYPOINTS = 33    # MediaPipe Pose hat 33 Keypoints
NUM_COORDINATES = 3   # x, y, z oder x, y, visibility
NUM_CLASSES = 3       # Pause, Wiederholung läuft, Wiederholung beendet
MODEL_TYPE = "LSTM"   # "LSTM" oder "TRANSFORMER"
HIDDEN_UNITS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Übungskonfigurationen
EXERCISE_TYPES = {
    "squat": {
        "name": "Kniebeuge",
        "error_types": ["nicht tief genug", "Knie zu weit vorne", "Rücken nicht gerade"]
    },
    "pushup": {
        "name": "Liegestütze",
        "error_types": ["nicht tief genug", "Körper nicht gerade", "zu schnell"]
    }
}

# Feedback-Konfiguration
FEEDBACK_DELAY = 1.5  # Sekunden nach Erkennung einer beendeten Wiederholung
AUDIO_FEEDBACK = True
TEXT_FEEDBACK = True

# Live-Stream-Konfiguration
USE_WEBCAM = True
CAMERA_ID = 0
DISPLAY_POSE = True
DISPLAY_FEEDBACK = True
