"""
detect_pneumonia module

This module provides functionality to detect pneumonia from chest X-ray images with transfer learning.
git Model: DenseNet121
"""

#import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

data_dir = "chest_xray"
img_size = (224,224)
batch_size = 32
class_mode = 'binary'

train_gen = train_datagen.flow_from_directory(
    os.path.join(data_dir,"train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode=class_mode,
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(data_dir,"train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode=class_mode,
    subset='validation'
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(data_dir,"test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=False
)

#visualize data
class_names = list(train_gen.class_indices.keys())
img, labels = next(train_gen)
plt.figure(figsize=(12,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(img[i])
    plt.title(class_names[int(labels[i])])
    plt.axis('off')
plt.show()

#model building
base_model = DenseNet121(weights='imagenet',
                         include_top=False,
                         input_shape=(224,224,3))
#freeze base model layers
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=pred)

#compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

print(model.summary())

#train model
epochs = 20
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[early_stop, model_checkpoint, reduce_lr],
    verbose=1
)

#evaluation
predict_probs = model.predict(test_gen)
pred_labels = (predict_probs > 0.5).astype(int).ravel()
true_labels = test_gen.classes

#confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

#plot training history
plt.figure(figsize=(12,5))
#accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

#loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

#save the trained model
model.save('pneumonia_detection_model.h5')
print("Model saved as pneumonia_detection_model.h5")