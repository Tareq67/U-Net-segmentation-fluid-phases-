# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:16:30 2021

@author: Eier
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 
import scipy.ndimage as ndimage
from matplotlib.widgets import Slider, Button, RadioButtons
import ipywidgets as ipyw
from math import ceil
import random
from volumentations import *

# Generate noise
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


def addGaussian(img, sigma):
    img = ndimage.gaussian_filter(img, sigma=sigma, order=0)
    return img


def getFluidDistribution(grid, cutOff):
   
    # Cutoff and set oil to be 254
    grid = np.where(grid >= cutOff, 254, grid)
    # Set glass bead to be 100
    grid = np.where(grid == 0, 100, grid)
    # Set water to be 255
    grid = np.where(grid < cutOff, 255, grid)
    # Set oil to be 0
    grid = np.where(grid == 254, 0, grid)
    return grid

def convertToClasses(grid):
    # Convert from [0,255] to [0,2]
    grid = np.where(grid == 100, 1, grid)
    grid = np.where(grid == 255, 2, grid)
    return grid

def saveModel(path, fileName, grid):
    with open(path + fileName, 'wb') as f:
        np.save(f, grid)
        
def openModel(path, fileName):
    with open(path+fileName, 'rb') as f:
        grid = np.load(f)
        return grid
    
def visualize(image, mask, vmin=0, vmax=255):
    fig, axes = plt.subplots(1,2, figsize=(15,15))
    ax = axes.flatten()
    ax[0].imshow(image, cmap = "gray",vmin=vmin, vmax=vmax)
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap = "gray",vmin=0, vmax=2)
    ax[1].set_title("Mask")

    
    


class ImageViewer:
    def __init__(self, volume, absoluteRange, figsize=(8,8), cmap='gray'):
        self.volume = volume
        self.absoluteRange = absoluteRange
        self.figsize = figsize
        self.cmap = cmap
        if absoluteRange:
            self.v = [0, 255]
        else:
            self.v = [np.min(volume), np.max(volume)]
        
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
        options=['x-y','y-z','z-x'], value='x-y',
        description='Slice plane selection:', disabled=False, 
        style={'description_width' : 'initial'}))
        
    def view_selection(self, view):
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y":[0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        ipyw.interact(self.plot_slice,
                         z = ipyw.IntSlider(min=0, max=maxZ, step=1, continous_update=False, description='Image Slice: '))
            
    def plot_slice(self, z):
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), vmin=self.v[0], vmax=self.v[1])

