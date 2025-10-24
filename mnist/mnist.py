import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img=x_train[0]
stages={"original":img}
eq_img=cv2.equalizeHist(img)
stages["equalized"]=eq_img
blur_img=cv2.GaussianBlur(eq_img,(5,5),0)
stages["blurred"]=blur_img
edges=cv2.Canny(blur_img,50,150)        
stages["edges"]=edges
fig,ax=plt.subplots(2,2,figsize=(6,6))
ax=ax.flat
for ax ,(title,im) in zip(ax,stages.items()):
    ax.imshow(im,cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("MNIST Image Processing Steps")
plt.tight_layout()
plt.show()


# preprocesing function
# Düzeltme: Fonksiyon artık tek bir resim (img) alır ve döngü kaldırıldı.
def preprocess_image(img):
    # img zaten tek bir 28x28 resimdir
    eq_img = cv2.equalizeHist(img)
    blur_img = cv2.GaussianBlur(eq_img, (5, 5), 0)
    edges = cv2.Canny(blur_img, 50, 150)
    # 28x28 görüntüyü 784 uzunluğunda düzleştirir
    features=edges.flatten()/255.0
    return features

num_train=1e4
num_test=1e3

# Düzeltme: Şimdi her bir x_train elemanına (tekil resim) düzeltilmiş fonksiyon uygulanıyor.
xtarin=np.array([preprocess_image(img) for img in x_train[:int(num_train)]])
xtest=np.array([preprocess_image(img) for img in x_test[:int(num_test)]])

ytrain=y_train[:int(num_train)]
ytest=y_test[:int(num_test)]

model=Sequential([
    # Input shape artık xtarin'in yeni boyutu (784) ile uyumlu.
    Dense(128,activation='relu',input_shape=(784,)),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
history=model.fit(xtarin,ytrain,epochs=10,batch_size=32,validation_split=0.2,verbose=2)
test_loss,test_acc=model.evaluate(xtest,ytest)
print(f"Test accuracy: {test_acc*100:.2f}%")


test_loss,test_acc=model.evaluate(xtest,ytest)
print(f"Test accuracy: {test_acc*100:.4f}%")
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()   
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Train Acc')
plt.plot(history.history['val_accuracy'],label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend() 