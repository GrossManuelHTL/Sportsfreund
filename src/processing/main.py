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
    visiblePoints = 0
    screenHeight = img.shape[0]
    screenWidth = img.shape[1]

    if lmList and len(lmList) > 19:
        length = lmList[19][2]
        per = np.interp(int(length), [157, 285], [100, 0])
        poseBar = np.interp(length, [157, 285], [150, 400])

        for point in lmList:
            x, y = point[1], point[2]
            visibility = point[3]

            if 0 <= x <= screenWidth and 0 <= y <= screenHeight and visibility > 0.5:
                visiblePoints += 1

        if visiblePoints < 33:
            cv2.putText(img, "Please move farther away", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    cv2.waitKey(1)