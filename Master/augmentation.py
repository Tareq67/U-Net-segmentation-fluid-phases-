# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:27:12 2021

@author: Ådne
"""
import numpy as np
import random
#import albumentations as A
from skimage.exposure import adjust_gamma
import cv2
from skimage.transform import resize
from volumentations import Compose, Flip, RandomRotate90, GaussianNoise, ElasticTransformPseudo2D
from albumentations.augmentations.functional import MAX_VALUES_BY_DTYPE, from_float, to_float

def resizeImgAndMask(img, mask, output_shape):
    img = resize(img, output_shape, order=1, mode='reflect', preserve_range=True)
    mask = resize(mask, output_shape, order=1, mode='reflect', preserve_range=True)
    mask = np.round(mask, decimals=0)
    return img, mask

def FlipImgs(patch_size):
    axis = random.randint(0,2)
    return Compose([
            Flip(axis, p=1)])
    
def RandomRotateImg(patch_size):
    return Compose([
            RandomRotate90((0,1), p=1)
            ])
    
def GaussianNoiseImg(patch_size):
    return Compose([
            GaussianNoise(var_limit=(0,0.03), p=1)
            ])
    
def RandomGammaImg(img):
    mn, mx = 0.8, 1.2
    gamma = random.uniform(mn, mx)
    img = adjust_gamma(img, gamma=gamma)
    return img

def ElasticTransformImg():
    return Compose([
            ElasticTransformPseudo2D(alpha=500, sigma=30, alpha_affine=1, p=1.0)
            ])

# The function below is a modified versions of the 2D Downscale function in the albumentations package
# Downscales the images, resizes them to original size to change the resolution.
def DownScale(img, mask):
    #Scale image down then scale back up
    scale_min = 0.25
    scale_max = 0.5
    scale = np.random.uniform(scale_min, scale_max)
    interpolation = cv2.INTER_NEAREST
    h, w, d = img.shape[0], img.shape[1], img.shape[2]
    # Process image
    need_cast = interpolation != cv2.INTER_NEAREST and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)
    downscaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    upscaled_img = cv2.resize(downscaled_img, (w,h), interpolation=interpolation)
    if need_cast:
        upscaled_img = from_float(np.clip(upscaled_img, 0, 1), dtype=np.dtype("uint8"))
    # Process mask
    need_cast = interpolation != cv2.INTER_NEAREST and mask.dtype == np.uint8
    if need_cast:
        mask = to_float(mask)
    downscaled_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=interpolation)
    upscaled_mask = cv2.resize(downscaled_mask, (w,h), interpolation=interpolation)
    if need_cast:
       upscaled_mask = from_float(np.clip(upscaled_mask, 0, 1), dtype=np.dtype("uint8"))
      
    return upscaled_img, upscaled_mask

def cropImg(img, mask):
    h, w, d, = img.shape[0], img.shape[1], img.shape[2]
    minDim, maxDim = 64, h-1
    dim = random.randint(minDim,maxDim)

    x = random.randint(0,w-dim)
    y = random.randint(0, h-dim)
    
    crop_img = img[y:y+dim, x:x+dim, :]
    crop_mask = mask[y:y+dim, x:x+dim, :]
    
    img = cv2.resize(crop_img, (w,h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(crop_mask, (w,h), interpolation=cv2.INTER_NEAREST)
    
    return img, mask

# Get augmentation. This is the function that is called by the image loader in the neural network training setup    
def getAugmentation(img, mask, augProb=0.5):
    data = {'image': img, 'mask':mask}
    shape = img.shape
    volumentationMethod = False
    
    # Calclate the upper bound of the random function in order to fulfill the augmentation probability
    num_augs = 7
    num_non_augs = (num_augs/augProb - num_augs)
    upper_bound = round(num_augs + num_non_augs - 1)
    # Decide at random which augmentaton method
    i = random.randint(0, upper_bound)
       
    # Methods
    if i == 0:
        aug = FlipImgs(shape)
        volumentationMethod = True
    elif i == 1:
        aug = RandomRotateImg(shape)
        volumentationMethod = True  
    elif i==2:
        aug = GaussianNoiseImg(shape)
        volumentationMethod = True
    elif i==3:
        img = RandomGammaImg(img)
    elif i==4:
        aug = ElasticTransformImg()
        volumentationMethod = True
    elif i==5:
        img, mask = DownScale(img, mask)
    elif i==6:
        img, mask = cropImg(img, mask)
            
    # Check if the volumentation package is used, and hence is need for processing the data
    if volumentationMethod == True:
        # Process data
        aug_data = aug(**data)
        img, mask = aug_data['image'], aug_data['mask']
        
    return img, mask
 
