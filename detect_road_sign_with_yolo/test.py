from ultralytics import YOLO
import cv2

model= YOLO("runs/detect/traffic-sign-model/weights/best.pt")   # load a custom trained model


img_path = "/Users/cd/computer_vision/detect_road_sign_with_yolo/traffic-sign-detection-yolov8/test/images/1_DSC8284_jpg.rf.36e88fbcf6f52dbf94a540b90f3a698d.jpg" # path to the test image
img=cv2.imread(img_path)  # read the image
results = model(img)  # predict the image
print(results)  # print results to console

for box in results.boxes:  # iterate through detected boxes
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # get box coordinates
    conf = box.conf[0]  # get confidence score
    cls = int(box.cls[0])  # get class id
    label = f"{model.names[cls]} conf:{conf:.2f}" # get class label


    # draw bounding box and label on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("Detected Road Signs", img)  # display the image with detections
cv2.waitKey(0).destroyAllWindows()   # wait for a key press and close the image window
cv2.imwrite("detect_road_sign_with_yolo/images/detected_image1.jpg", img)  # save the image with detections