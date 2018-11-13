# read input
import numpy as np
import os
from glob import glob

# neural network
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

import time

# # Building the network
# 
# Input patch size: 80x80
# 
# - conv: 78x78
# - conv: 76x76
# - pool: 38x38
# - conv: 36x36
# - conv: 34x34
# - conv: 32x32
# - pool: 16x16
# - conv: 14x14
# - conv: 12x12
# - pool: 6x6
# - conv: 4x4
# - conv: 2x2
# - conv (2x2): 1x1
# - FC: 1024
# - FC: 512
# - FC: 2
# 


# assuming an input of 80x80, 3 channels
def build_network(input_tensor, nonlinearity=lasagne.nonlinearities.rectify):
    network = L.InputLayer(shape=(None, 3, None, None), input_var=input_tensor)

    print 'Input shape', network.output_shape

    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 78x78
    
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    
    print 'Before M1', network.output_shape
    # 76x76

    network = L.MaxPool2DLayer(network, 2)
    
    print 'After M1', network.output_shape
    # 38x38
    
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 36x36
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 34x34
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 32x32
    
    print 'Before M2', network.output_shape
    network = L.MaxPool2DLayer(network, 2)
    print 'After M2', network.output_shape
    # 16x16
    
    network = L.Conv2DLayer(network, 256, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 14x14
    network = L.Conv2DLayer(network, 256, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 12x12
    
    print 'Before M3', network.output_shape
    network = L.MaxPool2DLayer(network, 2)
    print 'After M3', network.output_shape
    # 6x6
    
    network = L.Conv2DLayer(network, 512, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 4x4
    network = L.Conv2DLayer(network, 512, 3, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 2x2
    network = L.Conv2DLayer(network, 512, 2, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    # 1x1
    print 'Before fully conv', network.output_shape
    
    network = L.Conv2DLayer(network, 1024, 1, nonlinearity=nonlinearity)
    network = L.BatchNormLayer(network)
    network = L.Conv2DLayer(network, 512, 1, nonlinearity=nonlinearity)
    # 2 classes
    network = L.Conv2DLayer(network, 2, 1, nonlinearity=nonlinearity)    
    
    print 'Final output', network.output_shape
        
    return network

# assuming an input of 40x40, 3 channels
def build_network_small(input_tensor, nonlinearity=lasagne.nonlinearities.rectify):
    network = L.InputLayer(shape=(None, 3, None, None), input_var=input_tensor)

    print 'Input shape', network.output_shape

    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity, W=lasagne.init.GlorotUniform())
    # 38x38
    
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 36x36
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 34x34
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 32x32
    
    print 'Before M2', network.output_shape
    network = L.MaxPool2DLayer(network, 2)
    print 'After M2', network.output_shape
    # 16x16
    
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 14x14
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 12x12
    
    print 'Before M3', network.output_shape
    network = L.MaxPool2DLayer(network, 2)
    print 'After M3', network.output_shape
    # 6x6
    
    network = L.Conv2DLayer(network, 256, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 4x4
    network = L.Conv2DLayer(network, 256, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 2x2
    network = L.Conv2DLayer(network, 256, 2, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 1x1
    print 'Before fully conv', network.output_shape
    
    network = L.Conv2DLayer(network, 512, 1, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    network = L.Conv2DLayer(network, 256, 1, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    # 2 classes
    network = L.Conv2DLayer(network, 2, 1, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())    
    
    print 'Final output', network.output_shape
        
    return network


# assuming an input of 120x120x3 
def build_network_large(input_tensor, nonlinearity=lasagne.nonlinearities.rectify):
    network = L.InputLayer(shape=(None, 3, None, None), input_var=input_tensor)
    # 120x120
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 118x118
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 116x116
    network = L.MaxPool2DLayer(network, 2) # M1
    # 58x58
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 56x56
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 54x54
    network = L.Conv2DLayer(network, 64, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 52x52
    network = L.MaxPool2DLayer(network, 2) # M2
    # 26x26
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 24x24
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 22x22
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 20x20
    network = L.MaxPool2DLayer(network, 2) # M3
    # 10x10
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 8x8
    network = L.Conv2DLayer(network, 128, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 6x6
    network = L.Conv2DLayer(network, 256, 3, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 4x4
    network = L.Conv2DLayer(network, 512, 4, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    network = L.BatchNormLayer(network)
    # 1x1
    network = L.Conv2DLayer(network, 512, 1, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    # 1x1
    network = L.Conv2DLayer(network, 2, 1, nonlinearity=nonlinearity,W=lasagne.init.GlorotUniform())
    
    return network

def softmax(network):
    output = lasagne.layers.get_output(network)
    exp = T.exp(output - output.max(axis=1, keepdims=True)) #subtract max for numeric stability (overflow)
    return exp / exp.sum(axis=1, keepdims=True)

def softmax_deterministic(network):
    output = lasagne.layers.get_output(network, deterministic=True)
    exp = T.exp(output - output.max(axis=1, keepdims=True)) #subtract max for numeric stability (overflow)
    return exp / exp.sum(axis=1, keepdims=True)

def log_softmax(network):
    output = lasagne.layers.get_output(network)
    xdev = output - output.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def log_softmax_deterministic(network):
    output = lasagne.layers.get_output(network, deterministic=True)
    xdev = output - output.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)


def training_function(network, input_tensor, target_tensor, learning_rate, use_l2_regularization=True, l2_lambda=0.000001):
    # Get the network output and calculate metrics.
    network_output = softmax(network)
        
    if use_l2_regularization:
        l2_loss = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean() + l2_lambda * l2_loss
    else:
        loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean()
        
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), T.argmax(target_tensor,axis=1)), dtype=theano.config.floatX)
    
    # Get the network parameters and the update function.                      
    network_params = L.get_all_params(network, trainable=True)
    weight_updates = lasagne.updates.adam(loss, network_params, learning_rate=learning_rate)
    
    # Construct the training function.
    return theano.function([input_tensor, target_tensor], [loss, accuracy], updates=weight_updates)


def validate_function(network, input_tensor, target_tensor):
    # Get the network output and calculate metrics.
    network_output = softmax_deterministic(network)
    loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean()
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), T.argmax(target_tensor,axis=1)), dtype=theano.config.floatX)  
    
    # Construct the validation function.
    return theano.function([input_tensor, target_tensor], [loss, accuracy])

def evaluate_function(network, input_tensor):
    # Get the network output and calculate metrics.
    network_output = softmax_deterministic(network)
    
    # Construct the evaluation function.
    return theano.function([input_tensor], network_output)


params_file = './parameters-FCN-large.npz'

learning_rate = 1e-3

# Load network and its parameters
input_var = T.tensor4('inputs', dtype=theano.config.floatX)
target_var = T.tensor4('targets', dtype=theano.config.floatX)

# network = build_network(input_var, lasagne.nonlinearities.rectify)
# network = build_network_small(input_var, lasagne.nonlinearities.rectify)
network = build_network_large(input_var, lasagne.nonlinearities.rectify)
patch_size = 120

evaluation_fn = evaluate_function(network=network, input_tensor=input_var)

# load stored params
npz = np.load(params_file)
L.set_all_param_values(network, npz['params'])

threshold = 0.5
# compute probability maps (segmentations) on the test images
from PIL import Image

test_files_path = os.path.join('Test','*.jpg')
# test_files_path = os.path.join('Train', '*.jpg')
test_images = sorted(glob(test_files_path))

# set to True if we want some pretty plots for a presentation or whatever
plot = False
debug = False
save_output = True
save_directory = os.path.join('Train')
# save_directory = os.path.join('results', 'Train', 'probmaps')

segment_ids = np.arange(15407,18386)
segment_ids = np.concatenate((segment_ids,[10174]))

for img_id in segment_ids:
    tail = '{}.jpg'.format(img_id)
    
    img_fullpath = os.path.join('TrainSmall2','Train',tail)
    # img id (number)
    
    
    fname = 'probability-map-{}.npz'.format(img_id)
    if os.path.exists(os.path.join(save_directory, fname)):
        # skip if we made this one already
        continue
    
    print 'Loading image', tail
    img = Image.open(img_fullpath)
    img = np.array(img)
    img = img / 255.0
    if debug:
        print 'image id', img_id
        print('image shape {}'.format(img.shape))
        
    A, B, C = img.shape
    
    # divide the images in smaller segments to fit the GPUs memory
    A_STEP = A // 4
    B_STEP = B // 4
    
    # to store probability maps in
    probability_map = dict()
    
    stime = time.time()
    
    ia = 0
    for a in range(0, A, A_STEP):
        ib = 0
        for b in range(0, B, B_STEP):
            # take a BIG_STRIDE x BIG_STRIDE area of the image.
            if debug:
                print 'a', a, 'b', b
            img_slice = img[a:a+A_STEP, b:b+B_STEP, :].astype(np.float32)

            # reshape into 
            # channels, dimension1, dimension2
            cdd = img_slice.transpose( (2,0,1) )

            # bij gebrek aan betere ideeen, padding over 3 channels
            img_padded_r = np.pad(cdd[0,:,:], patch_size // 2, 'constant', constant_values=0)
            img_padded_g = np.pad(cdd[1,:,:], patch_size // 2, 'constant', constant_values=0)
            img_padded_b = np.pad(cdd[2,:,:], patch_size // 2, 'constant', constant_values=0)
    
            img_padded = np.array([img_padded_r, img_padded_g, img_padded_b])
            img_padded = np.expand_dims(img_padded, axis=0)
            
            if debug:
                print('computing probability map...')
            t = -time.time()
            # normal method
            probability = evaluation_fn(img_padded)
            preds = probability[0,1,:,:]
            t += time.time()
            if debug:
                print('computed probability map in {} seconds'.format(t))

            # collect the probability map in the lookup table
            probability_map[ (ia, ib) ] = preds
            
            ib += 1
        ia += 1
    
    etime = time.time()
    print 'it took {} seconds to compute the probability map'.format(etime - stime)
    
# simple upscale using Kronecker
# could be replaced by shift and stitch, but a perfect segmentation isn't really our goal
#             n = 8
#             preds = np.kron(preds, np.ones((n,n)))

    # extract the shape of the complete probability map
    shape = [0, 0]
    As = {}
    Bs = {}
    for key in probability_map.keys():
        a, b = key
        pms = probability_map[key].shape
        if not (a in As):
            shape[0] += pms[0]
            As[a] = True
        
        if not (b in Bs):
            shape[1] += pms[1]
            Bs[b] = True

    PMAP = np.zeros( (shape[0], shape[1]) )
    if debug:
        print 'probability map shape', PMAP.shape
    
    # order the keys
    keys = probability_map.keys()
    keys = sorted(keys, key=lambda t: (t[0], t[1]))
    if debug:
        print 'keys', keys
    
    # idea: iterate through keys, keep track of probability map coordinates for each value a/b
    saA = { 0:0 }
    saB = { 0:0 }
    for a, b in keys:
        # keys are sorted [(0,0), (0,1) ..]
        pm = probability_map[(a,b)]
        pms = pm.shape
        if not ((a+1) in saA):
            saA[(a+1)] = saA[a] + pms[0]
        if not ((b+1) in saB):
            saB[(b+1)] = saB[b] + pms[1]
        
        if debug:
            print 'a, b', a, b
            print 'PMAP[..].shape', PMAP[saA[a]:saA[a+1], saB[b]:saB[b+1]].shape
            print 'pms', pms
            
        PMAP[saA[a]:saA[a+1], saB[b]:saB[b+1]] = pm

    # save output
    if save_output:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # savez
        fname = 'probability-map-{}.npz'.format(img_id)
        np.savez(os.path.join(save_directory, fname), PMAP)

