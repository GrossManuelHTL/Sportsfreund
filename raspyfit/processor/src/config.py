SQUAT_SOURCE = "../samples/squats.mp4"

# config.py

# Exercise parameters
EXERCISE_PARAMS = {
    "squat": {
        "min_knee_angle": 90,
        "max_knee_angle": 170,
        "min_hip_angle": 80,
        "max_hip_angle": 170,
        "rep_threshold": 20,  # Angle difference to count as rep
    },
    "pushup": {
        "min_elbow_angle": 70,
        "max_elbow_angle": 160,
        "rep_threshold": 30,
    },
    "lunge": {
        "min_knee_angle": 90,
        "max_knee_angle": 170,
        "rep_threshold": 30,
    }
}

# Drawing parameters
COLORS = {
    "good": (0, 255, 0),    # Green
    "warning": (0, 255, 255),  # Yellow
    "bad": (0, 0, 255),     # Red
    "text": (255, 255, 255),  # White
    "angle": (255, 255, 0),  # Cyan
    "landmark": (0, 0, 255)  # Red
}

# Text parameters
FONT_SCALE = 0.7
TEXT_THICKNESS = 2