# -*- coding: utf-8 -*-
"""
Created on 2022
@author: Tareq

"""
import numpy as np
from PIL import Image

from pathlib import Path

from tqdm import tqdm

images_path = 'images/'
masks_path = 'masks/'

# 3D Numpy arrays path
path_data ="/content/Master/MyDrive/Master/test"
#path_data = '/content/Master/MyDrive/Master/output_images/SegementGrids'
images_3D = list((Path(path_data)/images_path).iterdir())
masks_3D = list((Path(path_data)/masks_path).iterdir())

path_data_2D = Path(path_data)/'2D'
if not path_data_2D.exists():
    path_data_2D.mkdir()
    (path_data_2D/images_path).mkdir()
    (path_data_2D/masks_path).mkdir()

# Loop over each 3D array
for img_path, mask_path in tqdm(zip(images_3D, masks_3D)):
    # Loop over each 2D slice
    img_3D, mask_3D = np.load(img_path), np.load(mask_path)
    for i, (img, mask) in enumerate(zip(img_3D, mask_3D)):
        # Extract 2D slice
        file_name = "slice_"+img_path.name.split('_')[-1][:-4]+"_"+str(i)
        print(np.bincount(mask.astype('uint8').flatten()))
        #break
        np.save(path_data_2D.as_posix()+'/'+images_path+file_name, img)
        np.save(path_data_2D.as_posix()+'/'+masks_path+file_name, mask)
