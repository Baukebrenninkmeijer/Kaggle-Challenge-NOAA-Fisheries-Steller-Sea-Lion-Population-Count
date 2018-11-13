#!/usr/bin/env python3

# general imports
import numpy as np
import matplotlib.pyplot as plt
import os

# keras specific imports
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

'''
conv(k, c, s)
with
    kernel size k
    stride s
    channels c

pool(k, s)
with
    kernel size k
    stride s

fc(c)
with
    outputs c

conv(11, 96, 4) : pool(3, 2) : conv(5, 256, 1) : fc(1024) : fc(2)

softmax the output

binary classification : object or not

'''


def build_network(LEARNING_RATE, SHAPE):
    print('DeepBox architecture')
    print('Assuming input of shape {}'.format(SHAPE))
    # initialize
    model = Sequential()

    model.add( Conv2D(64, (11, 11), padding='same', input_shape=SHAPE) )
    model.add( Activation('relu') )

    model.add( MaxPooling2D(pool_size=(3, 3), strides=(2,2)) )

    model.add( Conv2D(128, (5, 5), padding='same') )
    model.add( Activation('relu') )

    model.add( Flatten() )
    model.add( Dense(1024) )
    model.add( Activation('relu') )
    model.add( Dense(2) )
    model.add( Activation('softmax') )
    
    # initialize optimizer
    opt = keras.optimizers.rmsprop(lr=LEARNING_RATE, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # dummy training data
    batch_size = 10
    channels = 3
    X = np.random.random( (batch_size, channels, 64, 64) )

    SHAPE = X.shape[1:]
    LEARNING_RATE = 0.0001


    model = build_network(LEARNING_RATE, SHAPE)

    # om te trainen:
    # model.fit(X, y)
