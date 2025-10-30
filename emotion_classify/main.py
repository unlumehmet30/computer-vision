import cv2
import mediapipe as mp
import numpy as np

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Landmark index sets
LEFT_EYE = [159, 145]
LEFT_BROW = [65, 158, 159]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LOWER_LIP = 17
FACE_LEFT = 234
FACE_RIGHT = 454

# Convert normalized landmark to pixel coordinate
def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

# Detect emotion based on geometric features
def detect_emotion(landmarks, w, h):

    # Face width for normalization
    left_face = get_point(landmarks, FACE_LEFT, w, h)
    right_face = get_point(landmarks, FACE_RIGHT, w, h)
    face_width = np.linalg.norm(right_face - left_face)

    # Eyebrow to eye distance (average points)
    brow_avg = np.mean([get_point(landmarks, i, w, h) for i in LEFT_BROW], axis=0)
    eye_avg  = np.mean([get_point(landmarks, i, w, h) for i in LEFT_EYE], axis=0)
    brow_lift = np.linalg.norm(brow_avg - eye_avg) / face_width

    # Mouth width
    mouth_l = get_point(landmarks, MOUTH_LEFT, w, h)
    mouth_r = get_point(landmarks, MOUTH_RIGHT, w, h)
    mouth_width = np.linalg.norm(mouth_r - mouth_l) / face_width

    # Lip angle for sadness
    mid = (mouth_l + mouth_r) / 2
    lower_lip = get_point(landmarks, LOWER_LIP, w, h)
    lip_drop = (lower_lip[1] - mid[1]) / np.linalg.norm(mouth_r - mouth_l)

    # Emotion decision rules
    # Note: Rules are tuned based on normalized facial geometry
    if brow_lift > 0.085 and mouth_width > 0.32:
        return "Surprised"
    elif lip_drop > 0.18 and brow_lift < 0.07:
        return "Sad"
    elif brow_lift < 0.06 and mouth_width < 0.28:
        return "Angry"
    elif mouth_width > 0.36 and brow_lift > 0.055:
        return "Happy"
    else:
        return "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for fl in results.multi_face_landmarks:
            emotion = detect_emotion(fl.landmark, w, h)

            # Display emotion
            cv2.putText(frame, emotion, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Draw face mesh
            mp_drawing.draw_landmarks(
                frame, fl, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            )

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
