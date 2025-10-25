#import libraries
from tensorflow_datasets import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.data import AUTOTUNE
import tensorflow as tf
import matplotlib.pyplot as plt 
#veri seti yükleme
(ds_train,ds_test),ds_info=load("tf_flowers", split=["train[:80%]",
                          "train[80%:]"],
                        as_supervised=True,
                        with_info=True)

print(ds_info.features )
print("number of classes:", ds_info.features["label"].num_classes)




#örnek görselleri gösterme
figure = plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = figure.add_subplot(1, 3, i + 1)
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title("Label: {}".format(label.numpy()))
    ax.axis("off")
plt.tight_layout()
plt.show()


img_size = (128, 128)
#data augmentation+preprocessing
def preprocess_train(image, label):
    image = tf.image.resize(image, img_size)
    img=tf.image.random_flip_left_right(image)
    img=tf.image.random_brightness(img, max_delta=0.1)
    img=tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img=tf.image.random_crop(img, size=[100,100,3])
    img=tf.image.resize(img, img_size)
    img=tf.cast(img, tf.float32)/255.0
    return img,label
def preprocess_test(img, label):
    img = tf.image.resize(img, img_size)
    img = tf.cast(image, tf.float32) / 255.0
    return img, label 

#veri seti hazırlama
ds_train=(
        ds_train
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .batch(32)
        .prefetch(AUTOTUNE)
)   
ds_val=(ds_test
        .map(preprocess_test, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE))





#cnn model
model=Sequential([
    #feature extraction
    Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
    MaxPooling2D((2,2)),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    #classification
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(ds_info.features["label"].num_classes,activation="softmax")
])


#callbacks

early_stopping=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
reduce_lr=ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=3,min_lr=1e-6,verbose=1)  
model_checkpoint=ModelCheckpoint("best_model.h5",monitor="val_loss",save_best_only=True)





#derleme
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
print(model.summary())




#training
model.fit(ds_train,
          epochs=20,
          validation_data=ds_val,
          callbacks=[early_stopping,reduce_lr,model_checkpoint])






#evaluation
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model.history.history["accuracy"], label='Training Accuracy')
plt.plot(model.history.history["val_accuracy"], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()        
plt.subplot(1, 2, 2)
plt.plot(model.history.history["loss"], label='Training Loss')
plt.plot(model.history.history["val_loss"], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()