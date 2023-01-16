# -*- coding: utf-8 -*-
"""
Created on 2022
@author: Tareq

"""
import numpy as np
import matplotlib.pyplot as plt

path = "/Users/tareqaljamou/anaconda3/envs/Master/Data/ErodedGrids/25iterations/"
filename = "maxConCirc.npy"

with open(path+filename, 'rb') as f:
    grid = np.load(f)
print(grid.shape)

""" #show the maxonCirc grid before cutting off values
plt.imshow(grid[:,:,499]) 
plt.savefig('maxonCirc_beforeCutOff.pdf',bbox_inches='tight')
plt.show()   """
###########################################

cutoff_values = [5,7,10,15,20,25,27,30,35,37,40,41]
saturations=[]
pc_s=[]

for x in cutoff_values:
    SegGrid=np.copy(grid)
    SegGrid[grid>x] =2#oil
    SegGrid[grid<x] =1#water
    SegGrid[grid==0] =0 # solid
    pc=1/x
    pc_s.append(pc)#add pc to pc_s list 
    saturation=np.count_nonzero(SegGrid==1)/(np.count_nonzero(SegGrid==2)+np.count_nonzero(SegGrid==1))
    saturations.append(saturation)
    
    #save segment grids as .npy file
    with open("output_images/SegementGrids/masks/SegGrid_"+ str(x)+".npy", 'wb') as f:
        np.save(f,SegGrid)

    plt.figure()
    plt.imshow(SegGrid[:,:,250],vmin=0, vmax=2)
    segment_name = "output_images/SegementGrids/SegGrid"+ str(x) +".png"  
    plt.savefig(segment_name,bbox_inches='tight')
    plt.show()
    
plt.plot(saturations,pc_s)
plt.xlabel("Sw")
plt.ylabel("Pc")
plt.title("saturation vs capillary pressure")
plt.savefig('output_images/plot_Pc.pdf',bbox_inches='tight')
plt.show()



