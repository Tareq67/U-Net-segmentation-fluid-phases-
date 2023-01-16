# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:42:24 2021
@author: Ådne

Updated on 2022 
@author: Tareq
"""
from scipy import ndimage
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from augmentation import getAugmentation
import json
from nnUtils import process_image, process_mask
from UNet3DDropout import build_unet

# Define paths to the images and masks
path = "./Master/MyDrive/Master/output_images/SegementGrids/"
img_folder = "images/"
mask_folder = "masks/"
img_path = os.path.join(path, img_folder)
mask_path = os.path.join(path, mask_folder)

img_paths = [
        os.path.join(os.getcwd(), img_path, x)
        for x in os.listdir(img_path)]

mask_paths = [
        os.path.join(os.getcwd(), mask_path, x)
        for x in os.listdir(mask_path)]

print("Number of images:", len(img_paths))
print("Number of masks: ",len(mask_paths))

x_train = list(filter(lambda p: p.split('/')[-1].split('_')[1].split('.')[0] not in ['45', '34'], img_paths))
y_train = list(filter(lambda p: p.split('/')[-1].split('_')[1].split('.')[0] not in ['45', '34'], mask_paths))

x_val = list(filter(lambda p: p.split('/')[-1].split('_')[1].split('.')[0] in ['45', '34'], img_paths))
y_val = list(filter(lambda p: p.split('/')[-1].split('_')[1].split('.')[0] in ['45', '34'], mask_paths))
print("Number of training images:", len(x_train), "\nNumber of validation images:", len(x_val))

img = process_image(x_train[5])
print(img.shape)
plt.imshow(img[:,:,0])

mask = process_mask(y_train[5])
print(mask.shape)
plt.imshow(mask[:,:,0])


# Preprocessing functions
def train_preprocessing(x, y):  
    def f(x,y):
        x = x.decode()
        y = y.decode()
        x = process_image(x)
        y = process_mask(y)
        
        # On-line data augmentation
        x, y = getAugmentation(img=x, mask=y, augProb=0.5)
        return x.astype('float32'), y.astype('int32')
    
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    x = tf.expand_dims(x, axis=3)
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    return x,y

def val_preprocessing(x,y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        x = process_image(x)
        y = process_mask(y)
        return x, y
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    x = tf.expand_dims(x, axis=3)
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    return x,y
    
# Define data loaders
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Set batch size
batch_size = 1

# Load training and validation data
train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
        )

validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(val_preprocessing)
        .batch(batch_size)
        .prefetch(2))
    
# Load model
shape = (256,256,500,1)
num_classes = 3
modelName = "3D_UNet_256_200.h5"
model = build_unet(shape, num_classes)
   
# Set learning rate schedule
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

# Compile model
model.compile(
        loss="categorical_crossentropy",
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['categorical_accuracy']
        )

# Define callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(modelName, monitor="val_categorical_accuracy", verbose=1,save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

#Set number of epochs
epochs = 200

# Train model
history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb])

# Save training data in a json-file
with open(modelName.split('.')[0] + '.json', 'w') as file:

    json.dump(history.history, file)

##################

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.xlabel('epoch')
plt.ylabel('categorical_accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
