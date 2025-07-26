import cv2
import cvzone
from ultralytics import YOLO
import time

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    # time.sleep(1/30)  # to get the actual frame
    cv2.imshow('Car Counting',frame)
    if cv2.waitKey(1)== ord('x'):
        break
cv2.destroyAllWindows()
