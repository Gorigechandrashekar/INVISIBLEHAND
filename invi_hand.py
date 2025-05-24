import cv2 as cv
import numpy as np
import mediapipe as mp

cap=cv.VideoCapture(0)
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.7)

#Initialize background
background=None
frame_count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv.flip(frame,1)
    h, w, _=frame.shape

    #Capture background after 30 frames
    if background is None and frame_count>30:
        background=frame.copy()
        print("Background captured")
    frame_count+=1

    #Convert to RGB for MediaPipe
    rgb=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results=hands.process(rgb)

    if background is not None and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #Get bounding box of the hand
            x_coords=[]
            for lm in hand_landmarks.landmark:
                x_coords.append(lm.x)
            y_coords=[]
            for lm in hand_landmarks.landmark:
                y_coords.append(lm.y)
            
            x_min=int(min(x_coords)*w)-20
            x_max=int(max(x_coords)*w)+20
            y_min=int(min(y_coords)*h)-20
            y_max=int(max(y_coords)*h)+20

            #Boundaries check
            x_min=max(x_min,0)
            x_max=min(x_max,w)
            y_min=max(y_min,0)
            y_max=min(y_max,h)

            #Replace hand area with background
            frame[y_min:y_max, x_min:x_max]=background[y_min:y_max, x_min:x_max]

    cv.imshow('Invisible Hand', frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()
