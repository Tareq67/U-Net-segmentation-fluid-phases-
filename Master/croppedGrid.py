# -*- coding: utf-8 -*-
"""
Created on 2022
@author: Tareq

"""
import numpy as np
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

path = "/Users/tareqaljamou/anaconda3/envs/Master/Data/"
filename = "gridInitialPack.npy" 
with open(path+filename, 'rb') as f:
    grid = np.load(f)
    
print(grid.shape)
print(grid.shape[0]/2)
croppedSize=500 

croppedGrid = grid[int(grid.shape[0]/2-croppedSize/2):int(grid.shape[0]/2+croppedSize/2), int(grid.shape[0]/2-croppedSize/2):int(grid.shape[0]/2+croppedSize/2), int(grid.shape[0]/2-croppedSize/2):int(grid.shape[0]/2+croppedSize/2)]#CCC
print("Shape after crop",croppedGrid.shape)

with open(path+"CroppGrids/CroppedGrid500.npy", 'wb') as f:
    np.save(f,croppedGrid)
plt.imshow(croppedGrid[:,:,int(croppedSize/2)])
plt.savefig('output_images/CroppedGrid_'+str(croppedSize)+'.jpg',bbox_inches='tight')
plt.show()
