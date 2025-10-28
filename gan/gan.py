#import libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist


#data loading and preprocessing
(train_images,_),(_,_)=fashion_mnist.load_data()
train_images=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images=(train_images-127.5)/127.5 #normalize to [-1,1]
batch_size=256
buffer_size=60000
noise_dm=100
img_shape=(28,28,1)
epochs=2
train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

#generator model
def make_generator_model():
    model=tf.keras.Sequential()
    model.add(layers.Dense(7*7*256,use_bias=False,input_shape=(noise_dm,))) #first fully connected layer
    model.add(layers.BatchNormalization()) #normalization layer
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))

    return model
      #generator=make_generator_model()
#discriminator model
def make_discriminator_model():
    model=tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model    
    #discriminator=make_discriminator_model()
#loss functions and optimizers
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output,fake_output):
    real_loss=cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss=real_loss+fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output) 
generator=make_generator_model()
discriminator=make_discriminator_model()
generator_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

#supporting functions
seed=tf.random.normal([16,noise_dm])
def generate_and_save_images(model,epoch,test_input):
    predictions=model(test_input,training=False)
    fig=plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')
        plt.axis('off')
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    plt.savefig('generated_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

#training loop 
def train(dataset,epochs):
    for epoch in range(1,epochs+1):
        gen_total_loss=0
        disc_total_loss=0   
        batch_count=0


        for image_batch in dataset:
            noise=tf.random.normal([batch_size,noise_dm])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images=generator(noise,training=True)

                real_output=discriminator(image_batch,training=True)
                fake_output=discriminator(generated_images,training=True)

                gen_loss=generator_loss(fake_output)
                disc_loss=discriminator_loss(real_output,fake_output)

            gradients_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)
            gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
            gen_total_loss+=gen_loss
            disc_total_loss+=disc_loss
            batch_count+=1
        print('Epoch {}/{}: Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(epoch,epochs,gen_total_loss/batch_count,disc_total_loss/batch_count))
        generate_and_save_images(generator,epoch,seed)
train(train_dataset,epochs)

# #save models 