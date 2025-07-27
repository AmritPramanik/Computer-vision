# object detection manually by custom data
import torch
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO("../Yolo-Weights/yolov8l.pt").to(device)
print("YOLO is running on:", model.device)
cap = cv2.VideoCapture("../videos/bikes.mp4")

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

    # new_frame = cv2.flip(frame,1)
    cv2.imshow('YOLO V8', frame)
    if cv2.waitKey(1) == ord('x'):
        break

cv2.destroyAllWindows()