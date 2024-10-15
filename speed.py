import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Mouse callback function to get pixel positions
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# Set up window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Video capture setup
cap = cv2.VideoCapture('veh2.mp4')

# Read class names from the coco.txt file
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Object tracker initialization
tracker = Tracker()

# Define the lines (L1 and L2) for tracking vehicles
cy1 = 322
cy2 = 368
offset = 6

# Dictionaries and lists to track vehicle movement and counts
vh_down = {}
counter = []
vh_up = {}
counter1 = []

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for speed optimization
    count += 1
    if count % 3 != 0:
        continue

    # Resize the frame for easier handling
    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO model prediction on the frame
    results = model.predict(frame)
    boxes = results[0].boxes.data  # Extract the detection boxes
    px = pd.DataFrame(boxes).astype("float")

    # List to store the detected objects
    detections = []

    # Iterate over each detection
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        class_id = int(row[5])
        class_name = class_list[class_id]

        # Filter to only detect cars
        if 'car' in class_name:
            detections.append([x1, y1, x2, y2])

    # Update tracker with current detections
    bbox_id = tracker.update(detections)

    # Process each tracked object
    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        # Draw bounding box
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        # Check if vehicle crosses L1 going down
        if cy1 - offset < cy < cy1 + offset:
            vh_down[obj_id] = time.time()

        if obj_id in vh_down:
            if cy2 - offset < cy < cy2 + offset:
                elapsed_time = time.time() - vh_down[obj_id]
                if obj_id not in counter:
                    counter.append(obj_id)
                    distance = 10  # meters (change as per real-world setup)
                    speed_ms = distance / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(speed_kmh)) + ' Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Check if vehicle crosses L2 going up
        if cy2 - offset < cy < cy2 + offset:
            vh_up[obj_id] = time.time()

        if obj_id in vh_up:
            if cy1 - offset < cy < cy1 + offset:
                elapsed1_time = time.time() - vh_up[obj_id]
                if obj_id not in counter1:
                    counter1.append(obj_id)
                    distance1 = 10  # meters
                    speed_ms1 = distance1 / elapsed1_time
                    speed_kmh1 = speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(speed_kmh1)) + ' Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw lines and labels for L1 and L2
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Display vehicle counts
    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, 'Going down: ' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'Going up: ' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key
        break

cap.release()
cv2.destroyAllWindows()
