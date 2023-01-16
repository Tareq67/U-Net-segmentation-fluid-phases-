# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:18:11 2021

@author: Ådne
"""
import numpy as np 
import random
import albumentations as A

from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip,GaussNoise, Blur, Downscale, RandomContrast, RandomBrightness, RandomGamma
   
def get2DAugmentation(img_slice, mask_slice, augProb=0.5):
    # Change dtype
    img_slice = img_slice.astype('float32')
    mask_slice = mask_slice.astype('int32')
    
    # Calculate the number of indices needed to give the desired augmentation probability
    num_augs = 10
    num_non_augs = (num_augs/augProb - num_augs)
    upper_bound = round(num_augs + num_non_augs - 1)
    
    #Decide which, if any, augmentation
    augIdx = random.randint(0, upper_bound)
    
    if augIdx == 0:
        aug = RandomRotate90(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==1:
        aug = HorizontalFlip(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==2:
        aug = VerticalFlip(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==3:
        aug = GridDistortion(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==4:
        aug = GaussNoise(var_limit=(0.00001,0.0004), p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==5:
        aug = Blur(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==6:
        aug = Downscale(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==7:
        aug = RandomContrast(p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==8:
        aug = RandomBrightness(limit=0.02, p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
    elif augIdx==9:
        aug = RandomGamma(gamma_limit=(90,110), p=1.0)
        augmented = aug(image = img_slice, mask = mask_slice)
        img_slice = augmented["image"]
        mask_slice = augmented["mask"]
  
        
    return img_slice, mask_slice.astype('int32')

