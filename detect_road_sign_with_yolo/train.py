# import libraries
from ultralytics import YOLO
import os

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# train the model with custom dataset
model.train(
    data="traffic-sign-detection-yolov8/data.yaml",  # path to dataset
    epochs=1,            # number of epochs
    imgsz=640,             # image size
    batch=16,              # batch size
    name="yolov8n-traffic-sign-detection",  # experiment name
    lr0=0.001,             # initial learning rate (note: 'lr' → 'lr0')
    optimizer="SGD",       # optimizer type
    weight_decay=0.0005,   # weight decay
    momentum=0.9,          # momentum
    patience=5,            # early stopping patience
    workers=2,             # number of data loading workers
    device="cpu",          # "0" for GPU, "cpu" for CPU
    save=True,             # save final model
    save_period=1,         # save model every n epochs
    val=True,              # validate after each epoch
    verbose=True           # print training info
)
