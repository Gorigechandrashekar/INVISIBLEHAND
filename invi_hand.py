import cv2 as cv
import mediapipe as mp

class HandDetector():
    
    def __init__(self, mode = False, maxHands = 2 , detection = 0.5 , trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection = detection
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.Hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.Hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS) 

    def getposition(self,img,draw=True):
        li=[]
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id,lm in enumerate(handlms.landmark):
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    li.append([id,cx,cy])

        return li
