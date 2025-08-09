import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = ()
        if self.results and self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                xList, yList = [], []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = (xmin, ymin, xmax, ymax)

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20),
                                  (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)
            except IndexError:
                pass

        return self.lmList, bbox

    def fingersUp(self):
        """Returns a list of 5 integers: 1 if finger is up, 0 if down."""
        fingers = []
        if len(self.lmList) == 0:
            return [0, 0, 0, 0, 0]

        # Thumb
        if self.lmList[4][1] < self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        tips = [8, 12, 16, 20]
        for tip in tips:
            if self.lmList[tip][2] < self.lmList[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
