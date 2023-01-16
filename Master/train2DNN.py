# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:01:47 2021
@author: Ådne

Updated on 2022 
@author: Tareq
"""

from nnUtils import process_2D_image, process_mask
import os
import random
from sklearn.model_selection import train_test_split
from augmentation2D import get2DAugmentation
from tensorflow import keras
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold
import cv2
import tensorflow as tf
from UNet2DDropout import build_unet
from multiResUNet2D import MultiResUnet 
import json

# Define model name
modelName = "2DbuildUNet_exp5_256_50.h5"
# Define paths to images and masks
path = "./Master/MyDrive/Master/output_images/SegementGrids/2D/"
img_folder = "images/"
mask_folder = "masks/" 

img_path = os.path.join(path, img_folder) 
mask_path = os.path.join(path, mask_folder)
img_paths = sorted([
        os.path.join(os.getcwd(), img_path, x)
        for x in os.listdir(img_path)])
mask_paths = sorted([
        os.path.join(os.getcwd(), mask_path, x)
        for x in os.listdir(mask_path)])
#print(mask_paths)
print("Images:",len(img_paths))
print("Masks: ",len(mask_paths))

#define images number
set(p.split('/')[-1].split('_')[1] for p in img_paths)
#split train and validation sets
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = list(gkf.split(img_paths, mask_paths, groups=[p.split('/')[-1].split('_')[1] for p in img_paths]))[0]
select_every_frame = 2
x_train, x_val = [img_paths[idx] for idx in train_idx][::select_every_frame], [img_paths[idx] for idx in val_idx]
y_train, y_val = [mask_paths[idx] for idx in train_idx][::select_every_frame], [mask_paths[idx] for idx in val_idx]
print("x_train,", set(p.split('/')[-1].split('_')[1] for p in x_train))
print("Validation: ", set(p.split('/')[-1].split('_')[1] for p in x_val))
print("Number of training images:", len(x_train), "\nNumber of validation images:", len(x_val))

#visualize data after splitting
img = process_2D_image(x_train[8])
print(img.shape)
plt.imshow(img.squeeze(-1));

mask = process_mask(y_train[8])
print(mask.shape)
print(mask)
plt.imshow(mask)


#perform image augmentation on the training dataset only.
def train_preprocessing(x, y):
    def f(x,y):
        x = x.decode()
        y = y.decode()
        x = process_2D_image(x)
        y = process_mask(y)        
        
        x, y = get2DAugmentation(img_slice=x, mask_slice=y, augProb=0.5)
        return x, y
    #print('xxxxxx', x)
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    #print("xxxxx", x)
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    #print("yyyy", y)
    x.set_shape([H,W,1])
    y.set_shape([H,W,3])
    print("training shape",x.shape, y.shape)
    print("train x & y",x,y)
    return x,y

def val_preprocessing(x,y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        x = process_2D_image(x)
        y = process_mask(y)
        return x.astype('float32'), y.astype('int32') 
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    x.set_shape([H,W,1])
    y.set_shape([H,W,3])
    print("val shape",x.shape, y.shape)
    print("val x & y",x,y)
    return x,y

# Define data loaders   
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_loader= tf.data.Dataset
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#train_loader =<TensorSliceDataset shapes: ((), ()), types: (tf.string, tf.string)>
#validation_loader=<TensorSliceDataset shapes: ((), ()), types: (tf.string, tf.string)>
#print("train_loader",train_loader)

batch_size = 16
H, W = 256, 256

train_dataset = (train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2))
        
validation_dataset = (validation_loader.shuffle(len(x_val))
        .map(val_preprocessing)
        .batch(batch_size)
        .prefetch(2))

print("train_dataset",train_dataset)
print("validation_dataset",validation_dataset)

# Load model
shape = (256,256,1)
num_classes = 3    
model = build_unet(shape, num_classes)
#model = MultiResUnet(height=256, width=256, n_channels=1)

# Train model
initial_learning_rate = 1e-5
print(initial_learning_rate)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
       initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

model.compile(
        loss="categorical_crossentropy",
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['categorical_accuracy']
        )

# Define callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(modelName,verbose=1,save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
tensorboard_cb= tf.keras.callbacks.TensorBoard(log_dir="logs")

#print("STEPS_PER_EPOCH",STEPS_PER_EPOCH)
epochs = 50

print(tf. __version__)
print(np.array(train_dataset).ndim)

#"""
history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        callbacks= checkpoint_cb)


with open(modelName.split('.')[0] + '.json', 'w') as file:

    json.dump(history.history, file)
###############

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
