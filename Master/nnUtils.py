# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:59:36 2021

@author: Ådne
"""
import numpy as np
import matplotlib.pyplot as plt

import cv2

def read_file(filepath):
    with open(filepath, 'rb') as f:
        img = np.load(f)
        
    return img

def normalize(img):
    img = img / 255.0
    img = img.astype("float32")
    return img

def process_image(path):
    img = read_file(path)
    img = normalize(img)
    img = cv2.resize(img, dsize=(256, 256))
    return img

def process_2D_image(path):
    img = read_file(path)
    img = normalize(img)
    img = cv2.resize(img, dsize=(256, 256))
    #img = np.stack([img, img, img], axis=-1)
    img = np.expand_dims(img, axis=2)
    return img

def process_mask(path):
    mask = read_file(path)
    mask = cv2.resize(mask, dsize=(256, 256))
    mask = mask.astype("int32")
    return mask

    
def saveFile(path, fileName, grid):
    with open(path + fileName, 'wb') as f:
        np.save(f, grid)


def visualize(image, mask, vmin=0, vmax=1):
    fig, axes = plt.subplots(1,2, figsize=(15,15))
    ax = axes.flatten()
    ax[0].imshow(image, cmap = "gray",vmin=vmin, vmax=vmax)
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap = "gray",vmin=0, vmax=2)
    ax[1].set_title("Mask")
