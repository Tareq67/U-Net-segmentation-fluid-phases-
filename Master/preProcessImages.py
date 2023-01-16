# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:40:38 2021
@author: Ådne

Updated on 2022 by @author: Tareq

"""
import numpy as np
from glob import glob
import utilities
import scipy.ndimage as ndimage
from math import ceil
import matplotlib.pyplot as plt


# Function for adding noise. For noise size bigger than one, it creates a noise matrix with lower resolution than the image,
# before it scales it up and adds it to the image 
def addNoise(img, sigma, size=1):
    mu = 0
    shape = img.shape
    if size == 1:
        noise = np.random.normal(mu, sigma, shape)
    else:
        dims = (ceil(img.shape[0] / size),ceil(img.shape[0] / size),ceil(img.shape[0] / size))
        noise = np.random.normal(mu, sigma, dims)
        noise = noise.repeat(size, axis=0)
        noise = noise.repeat(size, axis=1)
        noise = noise.repeat(size, axis=2)
        noise = noise[:shape[0], :shape[1], :shape[2]]
        
    img = img + noise
    return img
# Funtion for adding gaussian blur
def addGaussian(img, sigma):
    img = ndimage.gaussian_filter(img, sigma=sigma, order=0)
    return img

# Preprocessing function
def preprocess(img, sigmaGaussian, sigmaNoise, sizeNoise, oilColor, solidColor, waterColor):
    if img.max() < 3:
        img = np.where(img == 0, oilColor, img)
        img = np.where(img == 1, solidColor, img)
        img = np.where(img == 2, waterColor, img)
    img = addGaussian(img, sigma=sigmaGaussian)
    img = addNoise(img, sigma=sigmaNoise, size = sizeNoise)
    img = addNoise(img, sigma=sigmaNoise/2, size=1)
    return img

def save_data(path, grid):
    with open(path, 'wb') as f:
        np.save(f, grid)

if __name__ == '__main__':
    # Parameters for preprocessing
    black = 51
    gray = 61
    white = 67
    sigmaG = 2
    sigmaN = 0.4
    sizeN = 2
    
    # Paths
    path = "/Users/tareqaljamou/anaconda3/envs/Master/output_images/SegementGrids/"
    maskFolder="masks/"
    imagesFolder = "images/"

    # Get paths to the masks
    masks = sorted(glob(path +  "masks/*"))
    
    # Preprocess all masks
    for maskPath in masks:
        print("maskpath",maskPath)        
        with open(maskPath, 'rb') as f:
            mask = np.load(f)
            print(mask.shape)
          
        image = preprocess(mask, sigmaGaussian=sigmaG, sigmaNoise=sigmaN, sizeNoise=sizeN, oilColor=black, solidColor=gray, waterColor=white)
        print("Finished preprocessing")       
        # Save to image folder
        utilities.saveModel(path=path + imagesFolder, fileName=maskPath.split('/')[-1], grid=image)#SegGrid_40.npy"
        plt.imshow(image[:,:,250])#####CCC
        plt.savefig(path + imagesFolder +maskPath.split('/')[-1]+'.jpg',bbox_inches='tight')
        plt.show()
   
        
