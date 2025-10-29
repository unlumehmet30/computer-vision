# car tracking  with yolo but without training


# Import necessary libraries
import cv2
from ultralytics import YOLO
import numpy as np

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Using the nano version for speed  
#video get
video_path = "IMG_5268.MOV" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# output video setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
# Initialize variables for tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.track(frame, 
                          persist=True,
                          conf=0.3,
                          iou=0.5,
                          tracker='bytetrack.yaml',
                          classes=[2]  # Class 2 corresponds to 'car' in COCO dataset
                          )
    annoted_frame = results[0].plot()

    cv2.imshow('Car Tracking', annoted_frame)
    out.write(annoted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

