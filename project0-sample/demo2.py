import cv2
from ultralytics import YOLO

model = YOLO("../Yolo-Weights/yolov8l.pt")
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO V8',annotated_frame)
    if cv2.waitKey(1) == ord('x'):
        break
cv2.destroyAllWindows()