# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:06:41 2017

@author: Timo
"""

import os
from glob import glob
import numpy as np
from skimage.feature import peak_local_max
from PIL import Image
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

from time import time

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 140, 140)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    
    model.add(Activation('softmax'))
    return model

def normalize_image(image):
    """
        Normalize an image per-channel.
    """
    means = np.mean(image, axis=(0,1))
    stds = np.std(image, axis=(0,1))
    return (image-means)/stds

# Create model & load weights
model = create_model()
model.load_weights(os.path.join('params_2','weights.02-0.79.hdf5'))

# Load images
test_maps_path = os.path.join('Test','*.npz')
test_maps = sorted(glob(test_maps_path))
map_ids = [int(os.path.splitext(m)[0].split('-')[-1]) for m in test_maps]

test_files = [os.path.join('Test','{}.jpg'.format(i)) for i in map_ids]

patch_size = 140

results = pd.DataFrame(columns=['test_id','adult_males','subadult_males','adult_females','juveniles','pups'])

classes = ['no_sealion','adult_males','subadult_males','adult_females','juveniles','pups']


for i in range(len(test_files)):
    img_id = map_ids[i]
    # We're doing 1/8th of the full test set, so after just append constant values
    if img_id > 2329:
        if img_id % 20 == 0:
            print('Classifying {} naively'.format(img_id))
        naive = {'test_id':img_id,
                 'adult_males':5,
                 'subadult_males':4,
                 'adult_females':26,
                 'juveniles':15,
                 'pups':11}
        results = results.append(naive,ignore_index=True)
        continue
    
    start = time()
    print('Classifying {}'.format(img_id))
    
    img_file = test_files[i]
    map_file = test_maps[i]
    
    # Load image
    image = np.array(Image.open(img_file))/255.0
    img_width = image.shape[0]
    img_height = image.shape[1]
    
    # Load segmentation
    segmentation = np.load(map_file)['arr_0']
    
    
    # Peak local max
    peaks = peak_local_max(segmentation, indices = True, threshold_abs=0.5)
    
    x_scale = img_width/segmentation.shape[0]
    y_scale = img_height/segmentation.shape[1]
    
    peaks_scaled = np.rint(peaks * [x_scale, y_scale]).astype(int)
    
    # Put the patches in one batch
    batch = np.zeros((len(peaks_scaled), 3, patch_size,patch_size))
    for j, coords in enumerate(peaks_scaled):
        x, y = coords[0], coords[1]
        
        offset = patch_size // 2
        xdiff = -(x-offset) if (x-offset < 0) else 0
        ydiff = -(y-offset) if (y-offset < 0) else 0
        patch = image[max(0,x-offset):min(x+offset,img_width), 
                      max(0,y-offset):min(y+offset,img_height), :]
        batch[j,:,xdiff:xdiff+patch.shape[0],ydiff:ydiff+patch.shape[1]] = patch.transpose((2,0,1))
    
    predictions = model.predict(batch)
    if len(predictions) == 0:
        # No sea lions found
        print('No sealions found')
        preds = {'test_id':img_id,
             'adult_males':0,
             'subadult_males':0,
             'adult_females':0,
             'juveniles':0,
             'pups':0}
        results = results.append(preds,ignore_index=True)
        continue
    
    class_predictions = np.argmax(predictions,axis=1)
    un,counts = np.unique(class_predictions,return_counts=True)
    preds = {'test_id':img_id,
             'adult_males':5,
             'subadult_males':4,
             'adult_females':26,
             'juveniles':15,
             'pups':11}
    for j,cl in enumerate(un):
        if cl == 0:
            continue
        preds[classes[cl]] = counts[j]
    results = results.append(preds,ignore_index=True)
    
    
    print("Time: %.2f s"%(time() - start))
results.to_csv('some_classified.csv',index=False)