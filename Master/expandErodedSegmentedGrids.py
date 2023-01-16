# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 08:50:10 2021
@author: Eier

Updated on 2022 by Tareq
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    with open(path, 'rb') as f:
        grid = np.load(f)
    return grid

def save_data(path, grid):
    with open(path, 'wb') as f:
        np.save(f, grid)


if __name__ == '__main__':
    
    erodedPath =  "/Users/tareqaljamou/anaconda3/envs/Master/output_images/SegementGrids/masks/"#with dist
    originalPath =  "/Users/tareqaljamou/anaconda3/envs/Master/Data/CroppGrids/"
    savePath =  "/Users/tareqaljamou/anaconda3/envs/Master/Data/ResizedErodedGrids/resized-seg/"
    
    cutOffVals = [23,27,29,30,31,32,33,34,35,36,37,38,40,41,45]
    
    for i in cutOffVals:
        # Load eroded grid with fluid distribution
        erodedFileName = "SegGrid_"+str(i)+".npy" 

        print(erodedFileName)
        erodedGrid = load_data(erodedPath+erodedFileName) 
        # Load original grid without distribution
        originalFileName = "CroppedGrid500.npy" 
        orgGrid = load_data(originalPath+originalFileName)
        
        print("Org grid - Min:{",orgGrid.min(),"}, Max: {",orgGrid.max(),"}")
        
        # Resize solids to original size
        dist = np.where(orgGrid==255, 0, erodedGrid)
        
        # Save data
        save_data(savePath+erodedFileName, dist)
        plt.imshow(dist[:,:,250])
        plt.savefig(savePath +'/Org_CroppedGrid500_'+str(i)+'.pdf',bbox_inches='tight')
        plt.show()
        
    
