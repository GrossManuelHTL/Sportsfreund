# test_pose.py
import cv2
import mediapipe as mp
import pyttsx3

# Text-to-Speech Setup
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Kamera öffnen
cap = cv2.VideoCapture(0)
pose = mp.solutions.pose.Pose()

while True:
    success, frame = cap.read()
    if not success:
        break

    # Pose erkennen
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # Feedback bei erkannter Pose
    if result.pose_landmarks:
        engine.say("Pose erkannt!")
        engine.runAndWait()

    # Anzeige (optional, kann am Pi weggelassen werden)
    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
