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


def build_cnn(input_var=None):
    in1 = lasagne.layers.InputLayer(shape=(None, 3, 120, 160),
                                        input_var=input_var)
    conv1 = lasagne.layers.Conv2DLayer(
            in1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    conv2 = lasagne.layers.Conv2DLayer(
            conv1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))
    conv3 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    conv4 = lasagne.layers.Conv2DLayer(
            conv3, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    pool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))
    unpool1 = UnpoolNoSwitchLayer(pool2, pool2)
    deconv1 = DeconvLayer(unpool1, conv4, lasagne.nonlinearities.rectify)
    deconv2 = DeconvLayer(deconv1, conv3, lasagne.nonlinearities.rectify)
    unpool2 = UnpoolNoSwitchLayer(deconv2, pool1)
    deconv3 = DeconvLayer(unpool2, conv2, lasagne.nonlinearities.rectify)
    deconv4 = DeconvLayer(deconv3, conv1, lasagne.nonlinearities.rectify)
    return deconv4


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

def main():
    def batch(generator, n = 1):
        while True:
            minibatch = []
            try:
                for j in range(n):
                    minibatch.append(generator.next())
                yield np.vstack(minibatch).reshape((-1, 3, 120, 160))
            except:
                if minibatch != []:
                    yield np.vstack(minibatch).reshape((-1, 3, 120, 160))
                raise StopIteration
        
    def fetch_image(addresses):
        for address in addresses:
            im = skio.imread(address).astype(theano.config.floatX)
            im = im[::4,::4,:]/255.
            yield im.flatten('F')
        
    def get_addresses(file):
        fp = open(file, 'r')
        lines = fp.readlines()
        fp.close()
        addresses = []
        for line in lines:
            lspl = line.split(' ')
            addresses.append(lspl[0])
        return addresses
    
    def threaded_generator(generator, num_cached=50):
        import Queue
        queue = Queue.Queue(maxsize=num_cached)
        sentinel = object()  # guaranteed unique reference
    
        # define producer (putting items into queue)
        def producer():
            for item in generator:
                queue.put(item)
            queue.put(sentinel)
    
        # start producer (in a background thread)
        import threading
        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()
    
        # run as consumer (read items from queue, in current thread)
        item = queue.get()
        while item is not sentinel:
            yield item
            queue.task_done()
            item = queue.get()
            
    num_epochs = 200
    # Training/validation images
    training_images = '/media/daniel/DATA/data_unencrypted/optic/txt/discs_train.txt'
    validation_images = '/media/daniel/DATA/data_unencrypted/optic/txt/discs_test.txt'
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    # Build net
    network = build_cnn(input_var)
    # Loss
    prediction = lasagne.layers.get_output(network, competition=True)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # Optimization
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    # Validation/Testing
    test_prediction = lasagne.layers.get_output(network, competition=False)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    # Compile functions
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)
    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # Load the dataset
        batch_size = 10
        train_addresses = get_addresses(training_images)
        train_generator = fetch_image(train_addresses)
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for minibatch in threaded_generator(batch(train_generator,batch_size)):
            train_err += train_fn(minibatch, minibatch)
            train_batches += 1
            print('B%i' % train_batches),
            sys.stdout.flush()

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        valid_addresses = get_addresses(validation_images)
        valid_generator = fetch_image(valid_addresses)
        for minibatch in threaded_generator(batch(valid_generator,batch_size)):
            val_err = val_fn(minibatch, minibatch)
            val_batches += 1
            print('V%i' % val_batches),
            sys.stdout.flush()

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # Optionally, you could now dump the network weights to a file like this:
    layers = lasagne.layers.get_all_param_values(network)
    for i, layer in enumerate(layers):
        np.savez('W'+str(i)+'.npz', layer)

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




if __name__ == '__main__':
    main()















