# object detection manually by custom data

import cv2
import cvzone
from ultralytics import YOLO
import math

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("../Yolo-Weights/yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(3,980)
cap.set(4,550)

while True:
    ret,frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # it for bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # it is for open CV
            # cv2.rectangle(frame,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,0),thickness=2)
            # It is for Cvzone
            w,h= x2-x1, y2 -y1
            cvzone.cornerRect(frame,(x1,y1,w,h))

            # for confident
            conf = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            cvzone.putTextRect(frame,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1-20)),scale=1,thickness=1)

    cv2.imshow('YOLO V8', frame)
    if cv2.waitKey(1) == ord('x'):
        break
cv2.destroyAllWindows()