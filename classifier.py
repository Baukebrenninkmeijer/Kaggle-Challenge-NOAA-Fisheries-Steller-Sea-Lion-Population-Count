
# coding: utf-8

# In[ ]:

from patchgenerator import PatchGenerator
import pandas as pd
import os
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import h5py
import datetime

# In[ ]:

data = pd.read_csv('correct_coordinates.csv', index_col=0, dtype={'y_coord': int, 'x_coord': int})
train_files_path = os.path.join('Train', '*.jpg')
train_files = sorted(glob(train_files_path))
# We split train/validation at id 727
split_idx = 727
coordinates_path = 'correct_coordinates.csv'
csv = pd.read_csv(coordinates_path,index_col=0)
# op deze manier gebruiken we alleen labels voor train_files op... 
files = dict()
# List of tuples (img, class, (row,col))
labels_train = []
labels_val = []
classes = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
for fullpath in train_files:
    prefix, fname = os.path.split(fullpath)
    name, ext = os.path.splitext(fname)
    file_nr = int(name)
    files[file_nr] = fullpath
    labels_nr = csv[csv['filename'] == fname]
    for _, row in labels_nr.iterrows():
        label_class = classes.index(row['category'])
        (x, y) = int(row['y_coord']), int(row['x_coord']) # logical?
        if file_nr < 727:
            labels_train.append( (file_nr, label_class, (x, y)) )
        else:
            labels_val.append((file_nr, label_class,(x,y)))
    # files, labels should be okay


# In[ ]:

# Very important: the file names should be passed in full to both generators
batch_size = 128
pgen_train = PatchGenerator(files, labels_train, batch_size, num_classes=len(classes)+1)
pgen_val = PatchGenerator(files,labels_val,batch_size, num_classes=len(classes)+1)

# In[ ]:

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('th')


# In[ ]:

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

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
print("???")


# In[ ]:

# I'm multiplying the loss for sea lions with a factor of 2
class_weight = {0:1.0, 1:2.0, 2:2.0, 3:2.0, 4:2.0, 5:2.0}

checkpoint = ModelCheckpoint('./params/weights.{epoch:02d}-{val_acc:.2f}.hdf5',monitor='val_acc',verbose=1,save_best_only=True,mode='max')
early_stop = EarlyStopping(min_delta=0.01,patience=3)
callbacks_list = [checkpoint,early_stop]
model.fit_generator(pgen_train, 600, 50,callbacks=callbacks_list,validation_data=pgen_val,validation_steps=50)
#1200, 50

# In[ ]:

model.save("wew.h5")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



