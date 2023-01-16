# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:16:56 2021
@author: Eier

Updated on 2022 by Tareq
"""
import os
from turtle import shape
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from glob import glob

# Functions
def erosion(grid, iterations):
    struct = ndimage.generate_binary_structure(3, 1)
    erodedGrid = ndimage.binary_erosion(grid, structure=struct, iterations=iterations, border_value=1)
    return erodedGrid

def showErosion(grid, erodedGrid, z=0):
    fig, axes = plt.subplots(1,2, figsize=(10,10))
    ax = axes.flatten()
    ax[0].imshow(grid[:,:,z], cmap='gray')
    ax[0].set_title("Original grid")
    ax[1].imshow(erodedGrid[:,:,z], cmap='gray')
    ax[1].set_title("Eroded grid")
    
if __name__ == '__main__':
    path = "/Users/tareqaljamou/anaconda3/envs/Master/Data/CroppGrids/" 
    filename = "CroppedGrid500.npy" 
    ########### CHANGE BELOW FOR DIFFERENT ITERATIONS ############################
    iterations = 25
    savePath = "/Users/tareqaljamou/anaconda3/envs/Master/Data/ErodedGrids/25iterations/"
    #############################################################################
    
    # Load filenames
    with open(path+filename, 'rb') as f:
        grid = np.load(f)
    print("Shape: ",grid.shape)
    binGrid = np.where(grid==255, 1, 0)
    print("binGrid",binGrid)
    erodedGrid = erosion(binGrid, iterations=iterations)
    showErosion(binGrid, erodedGrid, z=0)
    saveName = os.path.basename(os.path.normpath("eroded_"+filename))
    print("savename",saveName)
    print(saveName[:-4]) # to remove .npy form the name
    with open(savePath +saveName, 'wb') as f:
        np.save(f, erodedGrid)
    plt.imshow(erodedGrid[:,:,125])
    plt.savefig('output_images/'+str(iterations)+"iterations/"+saveName[:-4]+'.pdf',bbox_inches='tight')
    plt.show()
    
