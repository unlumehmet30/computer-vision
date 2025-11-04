# predict.py
"""
Webcam üzerinden canlı MNIST tahmin
ELİYLE ÇİZİLMİŞ RAKAMI ortadaki alandan okur ve ekrana yazdırır
"""

import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mnist_cnn.h5")

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

def preprocess(gray):
    # 200x200 ROI -> 28x28 MNIST boyutuna dönüştür
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype("float32") / 255.0

    # Model formatı: (1,28,28,1)
    gray = np.expand_dims(gray, axis=(-1, 0))
    return gray

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    box = min(h, w)//3
    cx, cy = w//2, h//2
    x1,y1 = cx-box//2, cy-box//2
    x2,y2 = x1+box, y1+box

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold (MNIST ile uyum yakalamak için)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    inp = preprocess(th)
    pred = model.predict(inp, verbose=0)
    digit = np.argmax(pred)
    conf = np.max(pred)

    # Kutuyu çiz ve sonucu yaz
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"{digit} ({conf*100:.1f}%)", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    # Sağ üst köşe küçük preview (28x28)
    preview = cv2.resize(th, (120,120), interpolation=cv2.INTER_NEAREST)
    frame[10:130, w-130:w-10] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Live MNIST Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
