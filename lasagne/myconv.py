#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

import os, sys, time

import lasagne
import numpy as np
import scipy.ndimage.measurements as spmeas
import skimage.io as skio
import theano
import theano.tensor as T

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    # Load CIFAR dataset
    print('Loading CIFAR 10')
    file = '/media/daniel/DATA/Cifar/cifar-10-batches-py/data_batch_'
    data = []
    labels = []
    for i in ['1','2','3','4']:
        data_dict = unpickle(file+i)
        data.append(data_dict['data'])
        labels.append(np.asarray(data_dict['labels']))
    X_train = np.vstack(data[:3])
    y_train = np.hstack(labels[:3])
    X_val = data[-1]
    y_val = labels[-1]
    data_dict = unpickle('/media/daniel/DATA/Cifar/cifar-10-batches-py/test_batch')
    X_test = np.asarray(data_dict['data'])
    y_test = np.asarray(data_dict['labels'])

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 3, 32, 32))/255.
    X_val = X_val.reshape((-1, 3, 32, 32))/255.
    X_test = X_test.reshape((-1, 3, 32, 32))/255.
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are int64, we cast them to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.


def build_cnn(weights, input_var=None):
    in1 = lasagne.layers.InputLayer(shape=(None, 3, 120, 160),
                                        input_var=input_var)
    conv1 = lasagne.layers.Conv2DLayer(
            in1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=weights[0], b=weights[1])
    conv2 = lasagne.layers.Conv2DLayer(
            conv1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=weights[2], b=weights[3])
    pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))
    conv3 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=weights[4], b=weights[5])
    conv4 = lasagne.layers.Conv2DLayer(
            conv3, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=weights[6], b=weights[7])
    return conv4


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def construct_cnn(weights):           
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    # Build net
    network = build_cnn(weights, input_var)
    # Loss
    prediction = lasagne.layers.get_output(network, competition=True)
    # Compile functions
    predict = theano.function([input_var], prediction)
    return predict

def load_weights(files):
    print('Loading weights')
    weights = []
    for file in files:
        data = np.load(file)
        weights.append(data['arr_0'])
    return weights

class DeconvLayer(lasagne.layers.Layer):
    def __init__(self, incoming, match, nonlinearity, **kwargs):
        super(DeconvLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.stride = match.stride
        self.output_shp = match.input_shape
        self.W = match.W.dimshuffle(1,0,2,3)[:,:,::-1,::-1]
        
    def get_output_for(self, input, **kwargs):
        shp = input.shape
        upsample = T.zeros((shp[0], shp[1], shp[2] * self.stride[0],
                            shp[3] * self.stride[1]), dtype=input.dtype)
        upsample = T.set_subtensor(upsample[:, :, ::self.stride[0],
                                            ::self.stride[1]], input)
        return T.nnet.conv2d(input, self.W, border_mode='full')

    def get_output_shape_for(self, input_shape):
        return self.output_shp
    
class NonmaxsuppressionLayer(lasagne.layers.Layer):     
    def get_output_for(self, input, competition=True, **kwargs):
        if competition == True:
            flat = input.flatten(ndim=3)
            maxes = T.argmax(flat,axis=2)
            mask = T.zeros(input.shape, dtype=input.dtype)
            flatmask = mask.flatten(ndim=3)
            T.set_subtensor(flatmask[:,:,maxes], 1)
            mask = T.reshape(flatmask, input.shape)
            return mask*input
        else:
            return input

class UnpoolNoSwitchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, match, **kwargs):
        super(UnpoolNoSwitchLayer, self).__init__(incoming, **kwargs)
        self.output_shp = match.input_shape
        self.stride = match.pool_size
    
    def get_output_for(self, input, **kwargs):
        return input.repeat(self.stride[0], axis=2).repeat(self.stride[1],
                                                           axis=3)
    
    def get_output_shape_for(self, input_shape):
        return self.output_shp

class CNNLocalize(object):
    def __init__(self):
        files = ['/home/daniel/Code/NN/lasagne/W0.npz',
                 '/home/daniel/Code/NN/lasagne/W1.npz',
                 '/home/daniel/Code/NN/lasagne/W2.npz',
                 '/home/daniel/Code/NN/lasagne/W3.npz',
                 '/home/daniel/Code/NN/lasagne/W4.npz',
                 '/home/daniel/Code/NN/lasagne/W5.npz',
                 '/home/daniel/Code/NN/lasagne/W6.npz',
                 '/home/daniel/Code/NN/lasagne/W7.npz',]
        weights = load_weights(files)
        self.predict = construct_cnn(weights)
    
    def center_activation(self, image):
        '''Find the Gaussian centre of the feature map activation'''
        image = image[::4,::4,:].reshape(-1, 3, 120, 160)
        preact = self.predict(image)
        cm = np.zeros((preact.shape[1],2))
        for i in np.arange(preact.shape[1]):
            cm[i,:] = spmeas.center_of_mass(preact[0,i,...])
        return cm
    
    def correlate_centres(self, source_file, results_file):
        '''Apply CNN to every image in file and return'''
        lines = self.get_lines(source_file)
        errors = np.zeros((32, 2, len(lines)))
        scale = [54., 74.]
        im_scale = np.asarray([480., 640.])
        for k, line in enumerate(lines):
            print('Image %i' % k)
            # Extract data
            lspl = line.split(' ')
            image_address = lspl[0]
            OD_j_true = int(lspl[1])    # 0-639
            OD_i_true = int(lspl[2])    # 0-479
            OD_true = np.asarray([OD_i_true, OD_j_true])*1.
            # Get image
            image = skio.imread(image_address)
            # Get activations
            cm = self.center_activation(image)
            errors[:,:,k] = cm*im_scale/scale - OD_true
            np.save(results_file, errors)
    
    def get_lines(self, source_file):
        '''Return lines from source file'''
        sf = open(source_file, 'r')
        lines = sf.readlines()
        sf.close()
        return lines
    
    def analyse(self, results_file):
        '''Analyse the results'''
        errors = np.load(results_file)
        dist = np.sum(errors**2,axis=1)
        dist = dist < 48**2
        RMSE = np.sqrt(np.mean(dist, axis=1))
        print np.amax(np.compress(1-np.isnan(RMSE), RMSE))
        
            
if __name__ == '__main__':
    source_file = '/media/daniel/DATA/data_unencrypted/optic/optic_disc_centers.txt'
    results_file = '/media/daniel/DATA/data_unencrypted/optic/results/CNNlocal.npy'
    cl = CNNLocalize()
    #cl.correlate_centres(source_file, results_file)
    cl.analyse(results_file)
    














