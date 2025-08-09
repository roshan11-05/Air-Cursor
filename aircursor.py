import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm

################################
wCam, hCam = 640, 480
frameR = 100          # Frame Reduction
smoothening = 5       # Cursor smoothing factor
################################

pTime = 0
plocX, plocY = 0, 0   # Previous location
clocX, clocY = 0, 0   # Current location
clickCooldown = 0.3   # Cooldown between clicks
lastClickTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # Get positions of index and thumb
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[4][1:]  # Thumb tip

        fingers = detector.fingersUp()

        # Convert coordinates to screen position
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Smooth the values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

        # -------- Pinch Detection for Click --------
        distance = np.hypot(x2 - x1, y2 - y1)

        # Dynamically adjust threshold based on camera size
        pinchThreshold = wCam * 0.04  # ~4% of camera width

        if distance < pinchThreshold:
            currentTime = time.time()
            if currentTime - lastClickTime > clickCooldown:
                autopy.mouse.click()
                lastClickTime = currentTime

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("AirCursor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
