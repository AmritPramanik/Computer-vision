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

model = YOLO("../Yolo-Weights/yolov8l.pt").to('cuda')
cap = cv2.VideoCapture("../videos/cars.mp4")
mask = cv2.imread('masking.jpg')

while True:
    ret,frame = cap.read()
    imgRegion = cv2.bitwise_and(frame,mask)
    results = model(imgRegion)
    annotated_frame = results[0].plot()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h),l=9)

            conf = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass=='car' or currentClass=="truck" or currentClass=='bus' or currentClass=="motorbike" and conf >0.5 :
                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1 - 20)), scale=1, thickness=1,
                               offset=3)



    cv2.imshow('YOLO-V8',frame)
    if cv2.waitKey(0) == ord('x'):
        break

cv2.destroyAllWindows()