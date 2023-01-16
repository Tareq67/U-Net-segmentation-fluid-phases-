# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:19:26 2021

@author: Eier
"""

# Utils package
# Carl Fredrik Berg, carl.f.berg@ntnu.no, 2017
#

import numpy as np

def readGrid(gridfile,dim,bin,npy):
    if bin==True:
        grid=np.fromfile(gridfile,dtype=np.uint8)
        grid=np.reshape(grid,dim)
    else:
        if npy==True:
            grid=np.load(gridfile)
        else:
            grid=np.loadtxt(gridfile)
            grid=np.reshape(grid,dim)

    return grid


def extendPotential(potential,direction,pin,pout):
    #Extend potential grid in flow direction
    extendedPotential=np.copy(potential)
    tangentialDim=np.array(np.shape(potential))
    tangentialDim[direction]=1
    tangentialDim=tuple(tangentialDim)
    pinM=np.copy(extendedPotential[0,:,:])
    pinM[np.invert(np.isnan(pinM))]=pin
    poutM=np.copy(extendedPotential[-1,:,:])
    poutM[np.invert(np.isnan(poutM))]=pout
    pinM=pinM.reshape(tangentialDim)
    poutM=poutM.reshape(tangentialDim)
    extendedPotential=np.append(pinM,extendedPotential,direction)
    extendedPotential=np.append(extendedPotential,poutM,direction)
    return extendedPotential


def calcFF(grid,direction,add):
    if direction==0:
        direction=2
    else:
        if direction==2:
            direction=0
    totalFF=0.0
    writeFile=open('FFVariations.txt','a')
    for ii in range(0,np.shape(grid)[direction]-2):
        if direction==0:
            diff=grid[ii,:,:]-grid[ii+1,:,:]
        if direction==1:
            diff=grid[:,ii,:]-grid[:,ii+1,:]
        if direction==2:
            diff=grid[:,:,ii]-grid[:,:,ii+1]
        diff[np.isnan(diff)]=0
        FF=np.size(diff)/((np.shape(grid)[direction]+add)*np.sum(diff))
        #print FF
        writeFile.write(str(FF)+'\n')
        totalFF+=FF
    writeFile.close()
    return totalFF/(np.shape(grid)[direction]-2)


def saveResult(grid,outfile,raw,plot):
    if raw:
        grid.tofile(outfile+'.raw')
    else:
        np.save(outfile,grid)

    if plot:
        import matplotlib.pyplot as plt
        dim=np.shape(grid)
        plt.imshow(grid[int(dim[0]/2)], interpolation='nearest')
        plt.show()


def global_index(dim, i, j, k):
    return i + dim[0] * (j + dim[1] * k)                    

def local_index(dim, n):
        k=n/(dim[0]*dim[1])
        n=n-k*dim[0]*dim[1]
        j=n/dim[0]
        i=n-j*dim[0]
        return i,j,k


def top_sort(graph,direction):
    graph_sorted=[]
    graph_unsorted=graph.copy()
    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    if direction==0:
        return list(reversed([item[0] for item in graph_sorted]))
    else:
        return list([item[0] for item in graph_sorted])

def dfs(graph, start):
    visited, stack = {}, [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited



def createPotentialGraph(potential,direction,outside):
    dim=np.shape(potential)
    graph={}
    graph[-1]=[]
    graph[np.size(potential)]=[]
    for ii in range(0,np.size(potential)):
        if potential[local_index(dim,ii)]>0:
            i,j,k=local_index(dim,ii)
            graph[ii]=[]
            if i>0:
                if(potential[i-1,j,k]>0 and potential[i-1,j,k]<potential[i,j,k]):
                    graph[ii]+=[global_index(dim, i-1, j, k)]
            if i<(dim[0]-1):
                if(potential[i+1,j,k]>0 and potential[i+1,j,k]<potential[i,j,k]):
                    graph[ii]+=[global_index(dim, i+1, j, k)]
            if j>0:
                if(potential[i,j-1,k]>0 and potential[i,j-1,k]<potential[i,j,k]):
                    graph[ii]+=[global_index(dim, i, j-1, k)]
            if j<(dim[1]-1):
                if(potential[i,j+1,k]>0 and potential[i,j+1,k]<potential[i,j,k]):
                    graph[ii]+=[global_index(dim, i, j+1, k)]
            if k>0:
                if(potential[i,j,k-1]>0 and potential[i,j,k-1]<potential[i,j,k]):
                    graph[ii]+=[global_index(dim, i, j, k-1)]
            if k<(dim[2]-1):
                if(potential[i,j,k+1]>0 and potential[i,j,k+1]<potential[i,j,k]):
                    graph[ii]+=[global_index(dim, i, j, k+1)]
            #add end-connections
            if outside==0:
                if direction==0:
                    if i==0:
                        graph[-1]+=[ii]
                    if i==(dim[0]-1):
                        graph[ii]+=[np.size(potential)]
                if direction==1:
                    if j==0:
                        graph[-1]+=[ii]
                    if j==(dim[1]-1):
                        graph[ii]+=[np.size(potential)]
                if direction==2:
                    if k==0:
                        graph[-1]+=[ii]
                    if k==(dim[2]-1):
                        graph[ii]+=[np.size(potential)]
            if outside==1:
                if direction==0:
                    if i==0:
                        graph[ii]+=[-1]
                    if i==(dim[0]-1):
                        graph[np.size(potential)]+=[ii]
                if direction==1:
                    if j==0:
                        graph[ii]+=[-1]
                    if j==(dim[1]-1):
                        graph[np.size(potential)]+=[ii]
                if direction==2:
                    if k==0:
                        graph[ii]+=[-1]
                    if k==(dim[2]-1):
                        graph[np.size(potential)]+=[ii]
                    
                    
    return graph



def transversingPotential(potential):
    #downStream=np.copy(potential)
    downStream=np.zeros(np.shape(potential),dtype='bool')
    downStream[:]=False
    #downStream[np.isnan(downStream)==False]=1
    downStream[:,:,0][(potential[:,:,0]<1) & (potential[:,:,0]>0)]=True
    for ii in range(0,np.shape(downStream)[2]-1):
        inNN=np.sum(downStream==True)
        downStream[:,:,ii+1][((potential[:,:,ii+1]-potential[:,:,ii])<0) & (downStream[:,:,ii]==True)]=True
        downStream[:,1:,ii+1][((potential[:,1:,ii+1]-potential[:,:-1,ii+1])<0) & (downStream[:,:-1,ii+1]==True)]=True
        downStream[:,:-1,ii+1][((potential[:,:-1,ii+1]-potential[:,1:,ii+1])<0) & (downStream[:,1:,ii+1]==True)]=True
        downStream[1:,:,ii+1][((potential[1:,:,ii+1]-potential[:-1,:,ii+1])<0) & (downStream[:-1,:,ii+1]==True)]=True
        downStream[:-1,:,ii+1][((potential[:-1,:,ii+1]-potential[1:,:,ii+1])<0) & (downStream[1:,:,ii+1]==True)]=True
        outNN=np.sum(downStream==True)
        changes=outNN-inNN
        print("First changes:", {changes})
    changes=1
    while changes>0:
        print(changes)
        #print downStream
        inNN=np.sum(downStream==True)
        #downstream
        downStream[:,:,1:][((potential[:,:,1:]-potential[:,:,:-1])<0) & (downStream[:,:,:-1]==True)]=True
        downStream[:,:,:-1][((potential[:,:,:-1]-potential[:,:,1:])<0) & (downStream[:,:,1:]==True)]=True
        downStream[:,1:,:][((potential[:,1:,:]-potential[:,:-1,:])<0) & (downStream[:,:-1,:]==True)]=True
        downStream[:,:-1,:][((potential[:,:-1,:]-potential[:,1:,:])<0) & (downStream[:,1:,:]==True)]=True
        downStream[1:,:,:][((potential[1:,:,:]-potential[:-1,:,:])<0) & (downStream[:-1,:,:]==True)]=True
        downStream[:-1,:,:][((potential[:-1,:,:]-potential[1:,:,:])<0) & (downStream[1:,:,:]==True)]=True
        outNN=np.sum(downStream==True)
        changes=outNN-inNN

    #####Upstream part
    upStream=np.zeros(np.shape(potential),dtype='bool')
    upStream[:]=False
    upStream[:,:,-1][(potential[:,:,-1]<1) & (potential[:,:,-1]>0)]=True
    for ii in range(np.shape(upStream)[2]-2,-1,-1):
        inNN=np.sum(upStream==True)
#        upStream[:,:,ii+1][((potential[:,:,ii+1]-potential[:,:,ii])<0) & (upStream[:,:,ii]==True)]=True
        upStream[:,:,ii][((potential[:,:,ii]-potential[:,:,ii+1])>0) & (upStream[:,:,ii+1]==True)]=True
        upStream[:,1:,ii][((potential[:,1:,ii]-potential[:,:-1,ii])>0) & (upStream[:,:-1,ii]==True)]=True
        upStream[:,:-1,ii][((potential[:,:-1,ii]-potential[:,1:,ii])>0) & (upStream[:,1:,ii]==True)]=True
        upStream[1:,:,ii][((potential[1:,:,ii]-potential[:-1,:,ii])>0) & (upStream[:-1,:,ii]==True)]=True
        upStream[:-1,:,ii][((potential[:-1,:,ii]-potential[1:,:,ii])>0) & (upStream[1:,:,ii]==True)]=True
        outNN=np.sum(upStream==True)
        changes=outNN-inNN
        print("First changes:", {changes})
    changes=1
    while changes>0:
        print(changes)
        inNN=np.sum(upStream==True)
        upStream[:,:,1:][((potential[:,:,1:]-potential[:,:,:-1])>0) & (upStream[:,:,:-1]==True)]=True
        upStream[:,:,:-1][((potential[:,:,:-1]-potential[:,:,1:])>0) & (upStream[:,:,1:]==True)]=True
        upStream[:,1:,:][((potential[:,1:,:]-potential[:,:-1,:])>0) & (upStream[:,:-1,:]==True)]=True
        upStream[:,:-1,:][((potential[:,:-1,:]-potential[:,1:,:])>0) & (upStream[:,1:,:]==True)]=True
        upStream[1:,:,:][((potential[1:,:,:]-potential[:-1,:,:])>0) & (upStream[:-1,:,:]==True)]=True
        upStream[:-1,:,:][((potential[:-1,:,:]-potential[1:,:,:])>0) & (upStream[1:,:,:]==True)]=True
        outNN=np.sum(upStream==True)
        changes=outNN-inNN

    transversingPot=np.copy(upStream)
    transversingPot[:]=False
    transversingPot[(downStream==True) & (upStream==True)]=True
    np.save('test.npy',downStream)
    np.save('test2.npy',upStream)

    reducedPot=np.copy(potential)
    reducedPot[transversingPot==False]=np.nan
    return reducedPot

def maxCircFunc(distGrid,grid,dim):
    distGrid=np.asarray(distGrid)
    grid=np.asarray(grid)
    maxRadius=int(np.max(distGrid))+1
    #print maxRadius
    distGrid=np.lib.pad(distGrid,maxRadius,'constant')
    grid=np.lib.pad(grid,maxRadius,'constant')
    values=[]
    index=[]
    count=0
    for ii in range(maxRadius,dim[0]+maxRadius):
        for jj in range(maxRadius,dim[1]+maxRadius):
            for kk in range(maxRadius,dim[2]+maxRadius):
                if grid[ii][jj][kk]==1:
                    values.append(distGrid[ii][jj][kk])
                    index.append((ii,jj,kk))
                    # position2index[ii,jj,kk]=count
                    count+=1
    
    values=np.asarray(values)
    index=np.asarray(index)
    sortedList = np.flip(np.argsort(values),0)
    

    low=np.array([0,0,0])
    high=np.array([dim[0],dim[1],dim[2]])
    maxCircIn=np.zeros(np.shape(grid),float)

    for nn in sortedList:
        radius=values[nn]
        intRadius=int(radius)+1
        coordinates=index[nn]
        #Create logical map for radius
        L = np.arange(-intRadius, intRadius + 1)
        X, Y, Z = np.meshgrid(L, L, L)
        logicalRadiusMap=np.array((X ** 2 + Y ** 2 + Z ** 2) <= radius ** 2)
        #print 'sum before', np.sum(maxCircIn)
        smallBox=maxCircIn[coordinates[0]-intRadius:coordinates[0]+intRadius+1,coordinates[1]-intRadius:coordinates[1]+intRadius+1,coordinates[2]-intRadius:coordinates[2]+intRadius+1]
        #print 'max before = ',np.max(smallBox)
        smallBox[logicalRadiusMap & (smallBox<radius)]=radius
        #print 'max after = ',np.max(smallBox)
        #print 'sum after', np.sum(maxCircIn)
        ##Continue!
        # directions=range(0,3)
        # lowBound=(coordinates[directions]-radius).astype(int)
        # lowBound[lowBound<0]=0
        # upBound=(coordinates[directions]+radius).astype(int)
        # upBound=np.minimum(upBound,dim)
        # for ii in range(0,3):
        #     if coordinates[ii]<radius:
        #         low[ii]=0
        #     else:
        #         low[ii]=int(coordinates[ii]-radius)
        #     if coordinates[ii]+radius+1>dim[ii]:
        #         high[ii]=dim[ii]
        #     else:
        #         high[ii]=int(coordinates[ii]+radius+1)
        # for ii in range(low[0],high[0]):
        #     for jj in range(low[1],high[1]):
        #         for kk in range(low[2],high[2]):
        #             if np.linalg.norm(coordinates-np.array([ii,jj,kk]))<radius and grid[ii][jj][kk]==1 and maxCircIn[ii,jj,kk]<radius:
        #                 maxCircIn[ii,jj,kk]=radius
    #print np.max(maxCircIn[maxRadius:-maxRadius,maxRadius:-maxRadius,maxRadius:-maxRadius])
    maxCircIn[grid==0]=0
    #print np.max(maxCircIn[maxRadius:-maxRadius,maxRadius:-maxRadius,maxRadius:-maxRadius])
    return maxCircIn[maxRadius:-maxRadius,maxRadius:-maxRadius,maxRadius:-maxRadius]


def artificalCompressible(potGrid,rounds):
    #this needs to take direction into the picture
    dims=np.shape(potGrid)
    potGridPad=np.lib.pad(potGrid,1,'constant',constant_values=np.nan)
    potGridPad[:,:,0]=1.0
    potGridPad[:,:,dims[2]+1]=0.0
    potGridPadCopy=np.copy(potGridPad)
    for ii in range(0,rounds):
        potGridPadCopy[1:dims[0],1:dims[1],1:dims[2]]=np.nanmean(np.array([potGridPad[0:dims[0]-1,1:dims[1],1:dims[2]],potGridPad[2:dims[0]+1,1:dims[1],1:dims[2]],potGridPad[1:dims[0],0:dims[1]-1,1:dims[2]],potGridPad[1:dims[0],2:dims[1]+1,1:dims[2]],potGridPad[1:dims[0],1:dims[1],0:dims[2]-1],potGridPad[1:dims[0],1:dims[1],2:dims[2]+1]]),axis=0)
        potGridPadCopy[np.isnan(potGridPad)]=np.nan
        potGridPad[1:dims[0],1:dims[1],1:dims[2]]=np.nanmean(np.array([potGridPadCopy[0:dims[0]-1,1:dims[1],1:dims[2]],potGridPadCopy[2:dims[0]+1,1:dims[1],1:dims[2]],potGridPadCopy[1:dims[0],0:dims[1]-1,1:dims[2]],potGridPadCopy[1:dims[0],2:dims[1]+1,1:dims[2]],potGridPadCopy[1:dims[0],1:dims[1],0:dims[2]-1],potGridPadCopy[1:dims[0],1:dims[1],2:dims[2]+1]]),axis=0)
        potGridPad[np.isnan(potGridPadCopy)]=np.nan
        #print np.sum((potGridPadCopy-potGridPad)**2)
    return potGridPad[1:dims[0]+1,1:dims[1]+1,1:dims[2]+1]


def createCylinderMask(gridDim,radius,start,stop):
	mask3d=np.zeros(gridDim, dtype=bool)
	z,y=np.ogrid[0:gridDim[2],0:gridDim[1]]
	zc=int(gridDim[2]/2.0)
	yc=int(gridDim[1]/2.0)
	mask2d=(z-zc)**2+(y-yc)**2 <= radius**2
	for ii in range(start,stop):
		mask3d[ii,:,:]=mask2d
	return mask3d

