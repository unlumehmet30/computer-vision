# ==========================
# U-Net Image Segmentation (Multi-Tile Dataset)
# ==========================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --------------------------
# 1. Veri Yükleme Fonksiyonu
# --------------------------
def load_data(root, img_size=(128, 128)):
    images, masks = [], []

    # Tüm alt klasörleri (ör: Tile 1, Tile 2, Tile 3) dolaş
    for tile_folder in os.listdir(root):
        tile_path = os.path.join(root, tile_folder)
        img_dir = os.path.join(tile_path, "images")
        mask_dir = os.path.join(tile_path, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"[Atlandı] {tile_path} içinde images/masks yok.")
            continue

        for file in os.listdir(img_dir):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(img_dir, file)
            mask_name = os.path.splitext(file)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"[Uyarı] Mask bulunamadı: {mask_path}")
                continue

            # Görüntü işle
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0

            # Maske işle
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = np.expand_dims(mask, axis=-1) / 255.0

            images.append(img)
            masks.append(mask)

    print(f"✅ Toplam {len(images)} görüntü ve {len(masks)} maske yüklendi.")
    return np.array(images, dtype="float32"), np.array(masks, dtype="float32")


# --------------------------
# 2. U-Net Model Tanımı
# --------------------------
def uNetModel(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# --------------------------
# 3. Eğitim Bloğu
# --------------------------
if __name__ == "__main__":
    x, y = load_data('areal_dataset')

    if len(x) == 0 or len(y) == 0:
        raise ValueError("❌ Veri seti boş! 'areal_dataset' dizin yapısını kontrol et.")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = uNetModel()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    callbacks = [
        keras.callbacks.ModelCheckpoint('unet_model.h5', save_best_only=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=8,
        callbacks=callbacks
    )

    # --------------------------
    # 4. Eğitim Grafiği
    # --------------------------
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.legend()
    plt.title("U-Net Eğitim Süreci")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
def predict_and_visualize(model, x_val, y_val, num_samples=3):
    preds = model.predict(x_val)

    for i in range(num_samples):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Giriş Görüntüsü")
        plt.imshow(x_val[i])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Gerçek Maske")
        plt.imshow(y_val[i].squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Tahmin Edilen Maske")
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.axis('off')

        plt.show()

    
predict_and_visualize(model, x_val, y_val)