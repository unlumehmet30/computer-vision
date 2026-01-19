import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError

df=pd.read_csv("dataset.csv")
 
image_folder='/Users/cd/Downloads/archive 2/boneage-training-dataset/boneage-training-dataset'
available_files=set(os.listdir(image_folder))
available_ids=set(f.replace(".png","") for f in available_files if f.endswith(".png"))
df=df[df["id"].astype(str).isin(available_ids)].reset_index(drop=True)
df["boneage"]=df["boneage"]/240.0
df["path"]=df["id"].apply(lambda x:os.path.join(image_folder,f"{x}.png"))


print(df.head())

plt.hist(df["boneage"]*240,bins=50)
plt.xlabel("kemik yasi")
plt.ylabel("frekans")
plt.title("kemik yasi distribution")
plt.tight_layout()
plt.show()

def load_image(df,img_size=128):
    images=[]
    valid_indices=[]
    for i,path in enumerate(df["path"]):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img=cv2.resize(img,(img_size,img_size))
        img=img/255.
        images.append(img)
        valid_indices.append(i)
    new_df=df.iloc[valid_indices].reset_index(drop=True)
    return np.array(images).reshape(-1,img_size,img_size),new_df["boneage"].values
X,y=load_image(df)
print(X.shape)
x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=.2,random_state=42)
datagen=ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=.3,
    width_shift_range=0.1,
    height_shift_range=.2

)
datagen.fit(x_train)
#cnn
model=Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(128,128,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))

#regression
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="linear"))


#compile
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="mae",
    metrics=[MeanAbsoluteError()]


)
callbacks=[
    EarlyStopping(patience=10,restore_best_weights=True,monitor="val_loss"),
    ModelCheckpoint("bone_age_model.keras",save_best_only=True,monitor="val_loss"),
    ReduceLROnPlateau(patience=5,factor=0.5,monitor="val_loss")
    
]


history=model.fit(
    datagen.flow(x_train,y_train,batch_size=32),
    validation_data=(x_val,y_val),
    epochs=5,
    callbacks=callbacks
)

plt.plot(history.history["loss"],label="train mae")
plt.plot(history.history["val_loss"],label="val mae")
plt.xlabel("epochs")
plt.ylabel("mae")
plt.title("training performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

preds=model.predict(x_val)*240
actuals=y_val*240
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_val[i].reshape(128,128),cmap="gray")
    plt.title(f"tahmin:{preds[i][0]:.0f}\ngercek:{actuals[i][0]:.0f}")
    plt.axis("off")
plt.title("bone age results")
plt.tight_layout()
plt.show()