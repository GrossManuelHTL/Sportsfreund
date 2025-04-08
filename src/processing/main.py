import cv2
import time
import PoseModule
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0
detector = PoseModule.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (960, 540))
    img = detector.findPose(img)
    lmList = detector.getPosition(img)

    if len(lmList) > 19:
        length = lmList[19][2]
        per = np.interp(int(length), [157, 285], [100, 0])
        poseBar = np.interp(length, [157, 285], [150, 400])


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    cv2.waitKey(1)
