#-*- coding: utf-8 -*- 
"""
Created on 2022
@author: Tareq

"""
import numpy as np
import matplotlib.pyplot as plt 
path = "/Users/tareqaljamou/anaconda3/envs/Master/Data/"
filename = "gridInitialPack.npy"  
with open(path+filename, 'rb') as f:
 grid = np.load(f) 
 print('b',grid.shape) 
  
#show the intial grid before cropping
plt.imshow(grid[:,:,1500]) 
plt.savefig('output_images/IntiailGrid1.pdf',bbox_inches='tight')
plt.show()  
