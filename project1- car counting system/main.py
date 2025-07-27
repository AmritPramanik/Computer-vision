import cv2
import cvzone
from ultralytics import YOLO
import time
import math

model = YOLO("../Yolo-Weights/yolov8l.pt").to('cuda')
cap = cv2.VideoCapture("../videos/cars.mp4")

while True:
    ret,frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()

    boxes = results

    for


    cv2.imshow('YOLO-V8',frame)
    if cv2.waitKey(1) == ord('x'):
        break

cv2.destroyAllWindows()