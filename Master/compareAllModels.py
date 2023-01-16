# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:26:54 2021
@author: Ådne

Updated on 2022 by Tareq
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
import cv2
#from skimage.filters import threshold_multiotsu
import pandas as pd
#from compareModelsUtils import extract_features
import matplotlib.gridspec as gridspec
import seaborn as sns
from nnUtils import process_mask, process_2D_image, process_image

def loadDeepLearningModel(path, modelName):
    model = keras.models.load_model(path + modelName)
    return model

def loadImg(filePath):
    with open(filePath, 'rb') as f:    
        img = np.load(f)
    return img
    
    
def plotAll(UNet2D, UNet3D, MultiResUNet, filePath, saveFig=False):
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(15,30))
    fig.tight_layout()
    gs = gridspec.GridSpec(4,2)
    gs.update(wspace=0.05)
    gs.update(hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])

    ax1.imshow(UNet2D, vmin=0, vmax=2)
    ax1.set_title("2D U-Net, synthetic data")
    ax2.imshow(UNet3D, vmin=0, vmax=2)
    ax2.set_title("3D U-Net, synthetic data")
    ax3.imshow(MultiResUNet, vmin=0, vmax=2)
    ax3.set_title("2D MultiRes U-Net, synthetic data")
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    if saveFig:
        plt.savefig(filePath)
        
def plotDiff(img, mask, prediction, difference, path, fileName):
    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(2,2, figsize=(15,15))
    ax = axes.flatten()
    ax[0].imshow(img, vmin=0, vmax=2)
    ax[0].set_title("Original image")
    ax[1].imshow(mask, vmin=0, vmax=2)
    ax[1].set_title("Mask")
    ax[2].imshow(prediction, vmin=0, vmax=2)
    ax[2].set_title("Prediction")
    ax[3].imshow(difference, vmin=0, vmax=2)
    ax[3].set_title("Difference from mask")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    plt.savefig(path + fileName)        
        
        
def plotProbabilities(UNet2D, UNet3D, MultiResUNet, filePath, saveFig=False):
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(15,30))
    fig.tight_layout()
    gs = gridspec.GridSpec(3,2)
    gs.update(wspace=0.05)
    gs.update(hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax1.imshow(UNet2D, vmin=0, vmax=2)
    ax1.set_title("2D U-Net Probabilities, synthetic data")
    ax2.imshow(UNet3D, vmin=0, vmax=2)
    ax2.set_title("3D U-Net Probabilities, synthetic data")
    ax3.imshow(MultiResUNet, vmin=0, vmax=2)
    ax3.set_title("2D MultiRes U-Net Probabilities, synthetic data")
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    if saveFig:
        plt.savefig(filePath)
        
        
def plotHeatMap(UNet2D, UNet3D, MultiResUNet, filePath, saveFig=False):
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(15,30))
    fig.tight_layout()
    gs = gridspec.GridSpec(3,2)
    gs.update(wspace=0.05)
    gs.update(hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    
    sns.heatmap(UNet2D, xticklabels=False, yticklabels=False, square=True, ax=ax1)
    sns.heatmap(UNet3D, xticklabels=False, yticklabels=False, square=True, ax=ax2)
    sns.heatmap(MultiResUNet, xticklabels=False, yticklabels=False, square=True, ax=ax3)

    #ax1.imshow(UNet2D, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("2D U-Net heatmap, synthetic data")
    #ax2.imshow(UNet3D, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("3D U-Net heatmap, synthetic data")
    #ax3.imshow(MultiResUNet, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("2D MultiRes U-Net heatmap, synthetic data")
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    if saveFig:
        plt.savefig(filePath)
    

imgPath2D =  "/content/Master/MyDrive/Master/output_images/SegementGrids/2D/images/"
imgName2D = "slice_37_307.npy"
imgPath3D = "/content/Master/MyDrive/Master/output_images/SegementGrids/images/"
imgName3D = "SegGrid_37.npy"
idx = 8

#Open images
img2D = loadImg(imgPath2D + imgName2D)
img2D = img2D.astype('uint8')
img3D = loadImg(imgPath3D + imgName3D)

print((img2D).min(), (img2D).max())

# 2D U-Net
UNet2D = loadDeepLearningModel("models/","C_2DbuildUNet_exp5_256_50.h5")
UNet2D_img = img2D / 255.0
print(UNet2D_img.min(), UNet2D_img.max())
UNet2D_img = UNet2D_img.astype('float32')
UNet2D_img =cv2.resize(UNet2D_img, (256,256))
UNet2D_pred = UNet2D.predict(np.expand_dims(UNet2D_img, axis=0))
UNet2D_pred = np.argmax(UNet2D_pred, axis=-1)

# 2D MultiRes U-Net
UNetMultiRes = loadDeepLearningModel("models/","C_2D_MultiResUNet_exp_256_50.h5")
UNetMultiRes_img = img2D / 255.0
print(UNetMultiRes_img.min(), UNetMultiRes_img.max())
UNetMultiRes_img = UNetMultiRes_img.astype('float32')
UNetMultiRes_img =cv2.resize(UNetMultiRes_img, (256,256))
UNetMultiRes_pred = UNetMultiRes.predict(np.expand_dims(UNetMultiRes_img, axis=0))
UNetMultiRes_pred = np.argmax(UNetMultiRes_pred, axis=-1)

# 3D U-Net
UNet3D = loadDeepLearningModel("models/","C_3D_UNet_256_200.h5")
img3D = img3D / 255.0
print(img3D.min(), img3D.max())
img3D = img3D.astype("float32")
img3D =cv2.resize(img3D, (256,256))
UNet3D_pred = UNet3D.predict(np.expand_dims(img3D, axis=0))
UNet3D_pred = np.argmax(UNet3D_pred, axis=-1)
UNet3D_pred_slice = UNet3D_pred[0,:,:,idx]

savePath ="/content/Master/MyDrive/Master/models/probabilityMap/"
saveNameAll = imgName2D.split('.')[0] + ".png"
saveNameHeat = imgName2D.split('.')[0] + "_heat.png"
saveNameProb = imgName2D.split('.')[0] + "_prob.png"
plotAll( UNet2D_pred[0,:,:], UNet3D_pred_slice, UNetMultiRes_pred[0,:,:], filePath=savePath+saveNameAll, saveFig=True)
plotHeatMap( UNet2D_pred[0,:,:], UNet3D_pred_slice, UNetMultiRes_pred[0,:,:], filePath=savePath+saveNameHeat, saveFig=True)
plotProbabilities( UNet2D_pred[0,:,:], UNet3D_pred_slice, UNetMultiRes_pred[0,:,:], filePath=savePath+saveNameProb, saveFig=True)

def read_image(path):
    with open(path, 'rb') as f:
        img = np.load(f)
    img = img / 255.0
    img = img.astype("float32")
    return img

def make_prediction(modelPath, modelName, path, img_name):
    model = keras.models.load_model(modelPath + modelName)
    img_path = path + "images/" + img_name
    img = read_image(img_path)
    print(f"Shape of image after loading: {img.shape}")
    img =cv2.resize(img, (256,256))
    p = model.predict(np.expand_dims(img, axis=0))
    p = np.argmax(p, axis=-1)
    print(f"Shape of prediction: {p.shape}")
    if p.ndim ==4:
        p = p[0,:,:,:]
    elif p.ndim == 3:
        p = p[0,:,:]
    p = p.astype(np.uint8)
    mask = process_mask(path + "masks/" + img_name)
    print(f"Shape of mask: {mask.shape}")
    
    return p, img, mask

def find_differences(p, mask):
    difference = cv2.absdiff(p.astype('uint8'), mask.astype('uint8'))
    return difference
    
def colorImgWithDifference(difference, img):
    # Make img and difference RGB
    img = img * 255.0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    print(f"Min rgb: {img_rgb.min()}, Max: {img_rgb.max()}")
    
    img_rgb[difference != 0] = [255,0,0]
    return img_rgb
      

# Make prediction for the 2d image
modelPath = "/content/Master/MyDrive/Master/models/"
path2D= "/content/Master/MyDrive/Master/test/2D/"
path3D= "/content/Master/MyDrive/Master/test/"
dSavePath = "/content/Master/MyDrive/Master/models/diff/"
imgName2D= "slice_36_100.npy"
imgName3D="SegGrid_36.npy"
p, img, mask = make_prediction(modelPath, "C_2DbuildUNet_exp5_256_50.h5", path2D,imgName2D)
difference = find_differences(p, mask)
img_rgb = colorImgWithDifference(difference, img)
saveName2D = imgName2D.split('.')[0] + "_2D.png"
plotDiff(img, mask, p, difference, dSavePath, saveName2D)


# Make prediction for the 2d res image
p, img, mask = make_prediction(modelPath, "C_2D_MultiResUNet_exp_256_50.h5", path2D, imgName2D)
difference = find_differences(p, mask)
img_rgb = colorImgWithDifference(difference, img)
saveName2DRes = imgName2D.split('.')[0] + "_2DRes.png"
plotDiff(img, mask, p, difference, dSavePath, saveName2DRes)


# Make prediction for the 3d image
p, img, mask = make_prediction(modelPath, "C_3D_UNet_256_200.h5", path3D, imgName3D)
sliceNr = 8
p = p[:,:,sliceNr]
img = img[:,:,sliceNr]
mask = mask[:,:,sliceNr]
difference = find_differences(p, mask)
img_rgb = colorImgWithDifference(difference, img)
saveName3D = imgName3D.split('.')[0] + "_3D.png"
plotDiff(img, mask, p, difference, dSavePath, saveName3D)



