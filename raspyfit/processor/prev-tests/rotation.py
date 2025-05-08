import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 3D-Koordinaten für Schultern und Hüften
            left_shoulder = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
            right_shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])

            # Richtungsvektor zwischen den Schultern
            shoulder_vec = right_shoulder - left_shoulder

            # Rotation berechnen: z. B. Winkel zwischen Schultern und Kameraebene
            angle = np.arctan2(shoulder_vec[2], shoulder_vec[0])  # Z-Achse vs X-Achse
            degrees = np.degrees(angle)

            cv2.putText(image, f"Rotation: {degrees:.2f} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Zeichne Pose
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Rotation", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
