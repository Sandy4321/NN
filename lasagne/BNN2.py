'''BNN'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import lasagne
import math
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from lasagne import utils
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset(dataset='MNIST'):
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
    if dataset == 'MNIST':
        # We'll now download the MNIST dataset if it is not yet available.
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            print("Downloading MNIST dataset...")
            urlretrieve(url, filename)
    
        # We'll then load and unpickle the file.
        import gzip
        with gzip.open(filename, 'rb') as f:
            data = pickle_load(f, encoding='latin-1')
    
        # The MNIST dataset we have here consists of six numpy arrays:
        # Inputs and targets for the training set, validation set and test set.
        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]
    
        # The inputs come as vectors, we reshape them to monochrome 2D images,
        # according to the shape convention: (examples, channels, rows, columns)
        X_train = X_train.reshape((-1, 1, 28, 28))
        X_val = X_val.reshape((-1, 1, 28, 28))
        X_test = X_test.reshape((-1, 1, 28, 28))
    
        # The targets are int64, we cast them to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_val = y_val.astype(np.uint8)
        y_test = y_test.astype(np.uint8)
    
        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'CIFAR10':
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

def build_bnn(input_var=None, masks=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = GaussianLayer(l_in, num_units=800,
                           name='l_hid1', prior_std=0.707,
                           nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = GaussianLayer(l_hid1, num_units=800,
                           name='l_hid2', prior_std=0.707,
                           nonlinearity=lasagne.nonlinearities.rectify)
    l_out = GaussianLayer(l_hid2, num_units=10,
                          name='l_out', prior_std=0.707,
                          nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def build_fullbnn(input_var=None, masks=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = FullGaussianLayer(l_in, num_units=800,
                               name='l_hid1', prior_std=0.5,
                               nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = FullGaussianLayer(l_hid1, num_units=800,
                               name='l_hid2', prior_std=0.707,
                               nonlinearity=lasagne.nonlinearities.rectify)
    l_out = FullGaussianLayer(l_hid2, num_units=10,
                              name='l_out', prior_std=0.707,
                              nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def build_bn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    network = lasagne.layers.DenseLayer(
            network, num_units=800, W=lasagne.init.GlorotUniform(),
            nonlinearity = lasagne.nonlinearities.linear)
    network = BayesBatchNormalizationLayer(network, lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            network, num_units=800, W=lasagne.init.GlorotUniform(),
            nonlinearity = lasagne.nonlinearities.linear)
    network = BayesBatchNormalizationLayer(network, lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            network, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return network

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform())
    network = BayesBatchNormalizationLayer(network, lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform())
    network = BayesBatchNormalizationLayer(network, lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform())
    network = BayesBatchNormalizationLayer(network, lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.DenseLayer(network,
            num_units=10, W=lasagne.init.GlorotUniform())
    network = BayesBatchNormalizationLayer(network, lasagne.nonlinearities.softmax)
    return network


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

def main(model='mlp', num_epochs=100, file_name=None, proportion=0.,
         save_name='./models/model.npz', dataset='MNIST', L2Radius=3.87,
         base_lr=0.0003):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset)
    dataset_size = X_train.shape[0]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'bnn':
        network = build_bnn(input_var)
    elif model == 'fullbnn':
        network = build_fullbnn(input_var)
    elif model == 'bn':
        network = build_bn(input_var)
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    batch_size = 500
    margin_lr = 25
    prediction = lasagne.layers.get_output(network, deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.sum()
    # We could add some weight decay as well here, see lasagne.regularization.
    
    reg = 0
    for layer in lasagne.layers.get_all_layers(network):
        if hasattr(layer, 'layer_type'):
            if layer.layer_type == 'GaussianLayer':
                reg += GaussianRegulariser(layer.M, layer.S,
                                          layer.prior_std, prior='Uniform')
            if layer.layer_type == 'FullGaussianLayer':
                reg += FullGaussianRegulariser(layer.W, layer.M, layer.S,
                                           layer.prior_std, prior='Uniform')
    loss = loss + reg/T.ceil(dataset_size/batch_size)
    
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    
    for layer in lasagne.layers.get_all_layers(network):
    #    if hasattr(layer, 'W'):
    #        layer.M = lasagne.updates.norm_constraint(layer.W, L2Radius)
        if hasattr(layer, 'M'):
            layer.M = lasagne.updates.norm_constraint(layer.M, L2Radius)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate = T.fscalar('learning_rate')
    updates = nesterov_momentum(loss, params, learning_rate=learning_rate,
                                momentum=0.9)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=False)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, learning_rate],
        loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    fig, ax, background = plot_progress()
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    v_err = []
    for epoch in range(num_epochs):
        learning_rate = get_learning_rate(epoch, margin_lr, base_lr)
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        i = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets, learning_rate=learning_rate)
            train_batches += 1
            i+=1
            #print('B%i' % i),
            #sys.stdout.flush()
        
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            v_err.append(err)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        ### PLOTTING ###
        points = ax.plot(v_err[-1000::], 'b')[0]
        fig.canvas.restore_region(background)
        ax.draw_artist(points)
        fig.canvas.blit(ax.bbox)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    save_model(network, save_name)
    print('Complete')
    return test_acc / test_batches * 100
    plt.close(fig)
    
def get_learning_rate(epoch, margin, base):
    return base*margin/np.maximum(6*epoch,margin)

def save_model(model, file_name):
    '''Save the model parameters'''
    print('Saving model..')
    params = {}
    for param in lasagne.layers.get_all_params(model):
        params[str(param)] = param.get_value()
    
    file = open(file_name, 'w')
    cPickle.dump(params, file, cPickle.HIGHEST_PROTOCOL)
    file.close()

# ##################### Custom layers #######################
class GaussianLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, nonlinearity,
                 M=None, R=None, prior_std=0.707, **kwargs):
        super(GaussianLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        if M is None:
            M = lasagne.init.Constant(0.0)
        if R is None:
            r = np.log(np.exp(np.sqrt(1./num_inputs))-1.)
            R = lasagne.init.Constant(r)
        self.M = self.add_param(M, (num_inputs+1, num_units), name='M')
        self.R = self.add_param(R, (num_inputs+1, num_units), name='R')
        self.S = T.log(1. + T.exp(self.R))
        self.nonlinearity = nonlinearity
        self.layer_type = 'GaussianLayer'
        self.prior_std = prior_std

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        b = T.ones_like(input[:,0]).dimshuffle(0,'x')
        X = T.concatenate([input,b],axis=1)
        if hasattr(self, 'mask'):
            M = T.dot(X,self.M*self.mask)
            s = T.sqrt(T.dot(X**2,self.mask*self.S**2))
        else:
            M = T.dot(X,self.M) 
            s = T.sqrt(T.dot(X**2,self.S**2))
        smrg = MRG_RandomStreams()
        E = smrg.normal(size=s.shape)
        H = M + s*E 
        # Nonlinearity
        return self.nonlinearity(H)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class FullGaussianLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, nonlinearity,
                 M=None, r=None, prior_std=0.707, **kwargs):
        super(FullGaussianLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        if M is None:
            M = lasagne.init.Constant(0.0)
        if r is None:
            r = np.log(np.exp(np.sqrt(1./num_inputs))-1.)
            r = lasagne.init.Constant(r)
        self.M = self.add_param(M, (num_inputs+1, num_units), name='M')
        self.R = self.add_param(r, (num_inputs+1, num_units), name='R')
        self.S = T.log(1. + T.exp(self.R))
        self.nonlinearity = nonlinearity
        self.layer_type = 'FullGaussianLayer'
        self.prior_std = prior_std

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        b = T.ones_like(input[:,0]).dimshuffle(0,'x')
        X = T.concatenate([input,b],axis=1)
        smrg = MRG_RandomStreams()
        self.W = self.M + smrg.normal(size=self.S.shape)*self.S
        H = T.dot(X,self.W)
        # Nonlinearity
        return self.nonlinearity(H)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class BatchNormalizationLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nonlinearity, g=None, b=None,
                 **kwargs):
        super(BatchNormalizationLayer, self).__init__(incoming, **kwargs)
        num_channels = self.input_shape[1]
        g = lasagne.init.Constant(1.)
        b = lasagne.init.Constant(0.)
        self.g = self.add_param(g, (num_channels,), name='g')
        self.b = self.add_param(b, (num_channels,), name='b')
        self.nonlinearity = nonlinearity
    
    def get_output_for(self, input, training=True, **kwargs):
        if input.ndim > 2:
            input2 = input.dimshuffle(1,0,2,3)
        elif input.ndim == 2:
            input2 = input.dimshuffle(1,0)
        else:
            print('Incompatible number of data dimensions')
        input2 = input2.flatten(2)
        mean = T.mean(input2, axis=1)
        var = T.var(input2, axis=1)
        eps = 1e-3
        if input.ndim > 2:
            norm_input = input - mean.dimshuffle('x',0,'x','x')
            norm_input = norm_input/T.sqrt(var + 1).dimshuffle('x',0,'x','x')
            output = self.g.dimshuffle('x',0,'x','x')*norm_input + \
                                                self.b.dimshuffle('x',0,'x','x')
        elif input.ndim == 2:
            norm_input = input - mean.dimshuffle('x',0)
            norm_input = norm_input/T.sqrt(var + 1).dimshuffle('x',0)
            output = self.g.dimshuffle('x',0)*norm_input + self.b.dimshuffle('x',0)
        return self.nonlinearity(output)

class BayesBatchNormalizationLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nonlinearity, g=None, b=None,
                 **kwargs):
        super(BayesBatchNormalizationLayer, self).__init__(incoming, **kwargs)
        num_channels = self.input_shape[1]
        g = lasagne.init.Constant(1.)
        b = lasagne.init.Constant(0.)
        self.g = self.add_param(g, (num_channels,), name='g')
        self.b = self.add_param(b, (num_channels,), name='b')
        self.nonlinearity = nonlinearity
    
    def get_output_for(self, input, training=True, **kwargs):
        if input.ndim > 2:
            input2 = input.dimshuffle(1,0,2,3)
        elif input.ndim == 2:
            input2 = input.dimshuffle(1,0)
        else:
            print('Incompatible number of data dimensions')
        input2 = input2.flatten(2)
        mean = T.mean(input2, axis=1)
        var = T.var(input2, axis=1)
        if input.ndim > 2:
            norm_input = input - mean.dimshuffle('x',0,'x','x')
            norm_input = norm_input*T.sqrt(1/(var + 1)).dimshuffle('x',0,'x','x')
            output = self.g.dimshuffle('x',0,'x','x')*norm_input + \
                                                self.b.dimshuffle('x',0,'x','x')
        elif input.ndim == 2:
            norm_input = input - mean.dimshuffle('x',0)
            norm_input = norm_input*T.sqrt(1/(var + 1)).dimshuffle('x',0)
            output = self.g.dimshuffle('x',0)*norm_input + self.b.dimshuffle('x',0)
        return self.nonlinearity(output)
    
# ##################### Custom regularisers #######################
def FullGaussianRegulariser(W, M, S, Sp, prior = 'Gaussian'):
    '''Return cost of W'''
    if prior == 'Gaussian':
        return -T.sum(T.log(S)) + 0.5*((T.sum(S**2)/(Sp**2)) - T.sum(M**2)/(Sp**2))
    elif prior == 'Uniform':
        return -T.sum(T.log(S)) 
    elif prior == 'GSM':
        return -T.sum(T.log(S)) - T.sum(T.log(0.25*Gaussian(W,0,0.367) +
                                              0.75*Gaussian(W,0,0.000912)))
    elif prior == 'Nothing':
        return 0.
    else:
        print('Invalid regulariser')
        sys.exit(1)

def GaussianRegulariser(M, S, Sp, prior = 'Gaussian'):
    '''Return cost of W'''
    if  prior == 'Uniform':
        return 0.#-T.sum(T.log(S)) 

def Gaussian(w, m, s):
    return T.exp(-0.5*((w-m)**2)/(s**2))/(s*T.sqrt(2*math.pi))


# ##################### Custom optimizations #######################

def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients"""
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form"""
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad #/ T.sqrt(T.maximum(T.var(grad),1))
    return updates

def apply_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including momentum
    Generates update expressions of the form"""
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x
    return updates

def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum
    Generates update expressions of the form"""
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_momentum(updates, momentum=momentum)

def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including Nesterov momentum
    Generates update expressions of the form"""
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]
    return updates

def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum
    Generates update expressions of the form"""
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)

# ################################### Plots ####################################
def plot_progress():
    """
    Display the simulation using matplotlib, optionally using blit for speed
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    #ax.set_xlim(0, 255)
    #ax.set_ylim(0, 255)
    ax.hold(True)

    plt.show(False)
    plt.draw()
    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)
    return fig, ax, background

# ################################### Main #####################################

if __name__ == '__main__':
    main(model='bnn', save_name='./models/mnistbnn.npz', dataset='MNIST',
         num_epochs=500, L2Radius=3.87, base_lr=1e-3)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    