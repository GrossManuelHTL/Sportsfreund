import cv2
import numpy as np
import tensorflow as tf

# Lade das MoveNet Lightning Modell
interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Funktion zum Zeichnen der Punkte
def draw_landmarks(image, keypoints):
    h, w, _ = image.shape
    for i in range(17):  # 17 Körperpunkte bei MoveNet
        y, x, score = keypoints[0, 0, i]
        if score > 0.3:  # nur bei sicherer Erkennung
            cx, cy = int(x * w), int(y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

# Kamera starten
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize & vorbereiten
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = tf.convert_to_tensor(img, dtype=tf.uint8)
    input_image = tf.expand_dims(input_image, axis=0)

    # Modell ausführen
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    # Landmarks auf Originalbild zeichnen
    draw_landmarks(frame, keypoints)

    cv2.imshow('MoveNet Pose Tracking', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
