# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:20:51 2021

@author: Carl Fredrik Berg, 2019
Some sections of the original script are removed as they were not needed for the invasion.
The reference to the paper that the script stems from can be found in the README.md.
"""
import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import heapq
import utils
from tqdm import tqdm

filePath = "/Users/tareqaljamou/anaconda3/envs/Master/Data/ErodedGrids/25iterations/"
#path = "/Users/tareqaljamou/anaconda3/envs/Master/Data/ErodedGrids/20iterations/withDist/"
fileName = "eroded_CroppedGrid500.npy"

with open(filePath + fileName, 'rb') as f:
    grid = np.load(f).astype(np.uint8)
    
dim = grid.shape
print('dim',dim)
saveNumpyArr = True
bin = 'True'

# #########################################
# Modified by Ådne
##########################################
grid[grid[:] == 1] = 255
###############################################
# 0 is solid and 1 is pore space
grid[grid[:] == 0] = 1
grid[grid[:] == 255] = 0
plt.imshow(grid[:,:,0], cmap='gray', vmin=0, vmax=2)
distanceTransform=ndi.distance_transform_edt(grid) 

grid[grid[:]==3]=0
grid[grid[:]!=0]=1
distanceTransform[grid[:]==0]=0

if saveNumpyArr:
	np.save(filePath + 'distTrans.npy',distanceTransform)

plt.figure(figsize=(5,5))
if dim[2]>1:
	#cross-section of 3D structure
	plt.imshow(np.ma.masked_where(grid[:,int(dim[1]/2),:]==0,distanceTransform[:,int(dim[1]/2),:]), interpolation='nearest',cmap='jet')
else:
	#2d structure
	plt.imshow(np.ma.masked_where(grid[:,:,0]==0,distanceTransform[:,:,0]), interpolation='nearest',cmap='jet')
plt.colorbar()
plt.savefig(filePath +'distanceTransform.pdf',bbox_inches='tight')

print('Start connected distance transform, modified Dijkstra algorithm')
connectedDistance=np.zeros(np.shape(distanceTransform),float)
curEdgeList=[]

for jj in tqdm(range(0,dim[1])):
	for kk in range(0,dim[2]):
		if grid[0][jj][kk]==1:
			edge=(0,jj,kk,distanceTransform[0,jj,kk])
			heapq.heappush(curEdgeList,(-edge[3],edge)) #as heap organize from smallest to biggest, we use the inverse for sorting

currentEdges=np.zeros(np.shape(distanceTransform),float)
currentEdges[:]=-2
currentEdges[:,:,:][np.where(grid[:,:,:]==0)]=-1
currentEdges[0,:,:][np.where(grid[0,:,:]==1)]=distanceTransform[0,:,:][np.where(grid[0,:,:]==1)]

def findSurounding(index,currentEdges):
	suroundingInd=[]
	for ii in range(0,3):
		testSorIndex=np.copy(index)
		testSorIndex[ii]-=1
		if testSorIndex[ii]>=0:
			if currentEdges[testSorIndex[0],testSorIndex[1],testSorIndex[2]]==-2:
				suroundingInd.append(testSorIndex.tolist())
		testSorIndex=np.copy(index)
		testSorIndex[ii]+=1
		if testSorIndex[ii]<dim[ii]:
			if currentEdges[testSorIndex[0],testSorIndex[1],testSorIndex[2]]==-2:
				suroundingInd.append(testSorIndex.tolist())
	return suroundingInd
tic=time.time()

while len(curEdgeList)>0:
    print(len(curEdgeList))
    edge=heapq.heappop(curEdgeList)[1]
    suroundingInd=findSurounding((edge[0],edge[1],edge[2]),currentEdges)
    for suroundEdge in suroundingInd:
        curDist=min(distanceTransform[suroundEdge[0],suroundEdge[1],suroundEdge[2]],edge[3])
        connectedDistance[suroundEdge[0],suroundEdge[1],suroundEdge[2]]=curDist
        currentEdges[suroundEdge[0],suroundEdge[1],suroundEdge[2]]=1
        heapq.heappush(curEdgeList,(-curDist,(suroundEdge[0],suroundEdge[1],suroundEdge[2],curDist)))
        currentEdges[edge[0],edge[1],edge[2]]=-1
toc=time.time()
print("Time while loop = ",{toc-tic})
print('Done connected distance transform')

maxConnectedCirc=utils.maxCircFunc(connectedDistance,grid,dim)
np.save(filePath + 'maxConCirc.npy',maxConnectedCirc)
