#import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp


#angle prediction
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


#mediapipe moduls
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap=cv2.VideoCapture("squat_test1.avi")

#rule based posture classification
counter = 0 
stage = None

def posture_classification(angle):
    if angle <100:
        return "squat"
    elif 100<=angle>=160:
        return "stand"
    else:
        return "standing"
print(calculate_angle((0,1),(0,0),(1,0)))    
print(posture_classification(150))


#posture 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = pose.process(img)
        img.flags.writeable = True

        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        try:
            landmaarks=results.pose_landmarks.landmark
            hip= [landmaarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmaarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee= [landmaarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                   landmaarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle= [landmaarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmaarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]   
            angle=calculate_angle(hip,knee,ankle)
            posture=posture_classification(angle)
            
            #count squats
            if angle < 90:
                stage = "down"
            if angle > 160 and stage == 'down':
                stage = "up"
                counter += 1
            cv2.putText(img, f"knee angle: {angle:.2f}", (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"squat count: {counter}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"posture: {posture}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        except:
            pass
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                      )
        cv2.imshow("Mediapipe Feed", img )
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



                
