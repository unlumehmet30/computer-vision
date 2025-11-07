import cv2
import numpy as np
from ultralytics import YOLO

# --- Yardımcı Fonksiyon ---
def get_line_side(x, y, line_start, line_end):
    # Noktanın çizginin hangi tarafında olduğunu belirler (+1, -1 veya 0 döner)
    x1, y1 = line_start
    x2, y2 = line_end
    return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))

# --- YOLO Modelini Yükle ---
model = YOLO("yolov8n.pt")

# --- Video Kaynağını Aç ---
cap = cv2.VideoCapture("IMG_5268.MOV")
success, frame = cap.read()
if not success:
    exit("Video açılamadı veya bulunamadı!")

frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
frame_height, frame_width = frame.shape[:2]

# --- ÇAPRAZ ÇİZGİ (Sol Alt → Sağ Üst) ---
line_start = (int(frame_width * 0.6), int(frame_height * 0.9))   # Sol alt köşeye yakın
line_end   = (int(frame_width * 0.9), int(frame_height * 0.4))   # Sağ üst köşeye yakın

# --- Sayım Değerleri ---
counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
counted_ids = set()
object_last_side = {}

# --- Ana Döngü ---
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    results = model.track(frame, persist=True, stream=False, conf=0.5, tracker="bytetrack.yaml")

    if results and len(results) > 0 and getattr(results[0].boxes, "id", None) is not None:
        ids = results[0].boxes.id.int().tolist()
        classes = results[0].boxes.cls.int().tolist()
        xyxy = results[0].boxes.xyxy

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]

            if class_name not in counts:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            current_side = get_line_side(cx, cy, line_start, line_end)
            prev_side = object_last_side.get(track_id, None)
            object_last_side[track_id] = current_side

            if prev_side is not None and prev_side != current_side:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    counts[class_name] += 1

            # --- Görselleştirme ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # --- ÇAPRAZ ÇİZGİYİ ÇİZ ---
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
    cv2.circle(frame, line_start, 6, (0, 255, 0), -1)
    cv2.circle(frame, line_end, 6, (0, 255, 0), -1)

    # --- Sayım Bilgisi Yazdırma ---
    y_offset = 30
    for cls, count in counts.items():
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    cv2.imshow("Car Tracking and Counting (Diagonal Line)", frame)

    # Çıkış için Q tuşu
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
