import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import math
from sort import *

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

model = YOLO("../Yolo-Weights/yolov8l.pt").to('cuda')
cap = cv2.VideoCapture("../videos/cars.mp4")
mask = cv2.imread('img/masking.jpg')

# tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
totalCount = []
limits = [160,380,690,380]

while True:
    ret,frame = cap.read()
    imgRegion = cv2.bitwise_and(frame,mask)

    imageGraphics = cv2.imread('img/graphics.png',cv2.IMREAD_UNCHANGED)
    imageGraphics = cv2.resize(imageGraphics,(imageGraphics.shape[1]//2,imageGraphics.shape[0]//2))
    frame = cvzone.overlayPNG(frame,imageGraphics,(0,0))

    results = model(imgRegion)
    annotated_frame = results[0].plot()

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass=='car' or currentClass=="truck" or currentClass=='bus' or currentClass=="motorbike" and conf >0.5 :
                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1 - 20)), scale=1, thickness=1,
                               offset=3)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=9,rt= 2,colorR=(0,0,255))

                concurrentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,concurrentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        print(id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l=9,rt= 2,colorR=(255,0,0))
        # cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1 - 20)), scale=1, thickness=2,
        #                    offset=5)

        cx,cy = x1+w//2 , y1+h//2
        cv2.circle(frame,(cx,cy),5,(0,255,0),-1)

        if limits[0]< cx <limits[2] and  limits[1] <= cy:
            print(id)
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255,0), 3)

    cvzone.putTextRect(frame, f'{len(totalCount)}', (4),offset=0)

    cv2.namedWindow('YOLO-V8', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO-V8', 940, 600)
    cv2.imshow('YOLO-V8',frame)
    if cv2.waitKey(1) == ord('x'):
        break

cv2.destroyAllWindows()