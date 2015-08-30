'''BNN'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import lasagne
import numpy as np
import theano
import theano.tensor as T

from matplotlib import pyplot as plt
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

def build_mlp(input_var=None, masks=None, temp=1):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_in, p=0.2), num_units=800,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid1, p=0.5), num_units=800,
            W=lasagne.init.GlorotUniform(), name='l_hid2')
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid2, p=0.5), num_units=10,
            W=lasagne.init.GlorotUniform(), name='l_out')
    l_soft = SoftermaxNonlinearity(l_out, temp=temp)
    return l_soft

def build_glp(input_var=None, masks=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_hid1_drop = GaussianDropoutLayer(l_hid1, prior_std=0.707,
            nonlinearity=lasagne.nonlinearities.rectify, name='l_hid1_drop')
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            W=lasagne.init.GlorotUniform(), name='l_hid2')
    l_hid2_drop = GaussianDropoutLayer(l_hid2, prior_std=0.707,  
            nonlinearity=lasagne.nonlinearities.rectify, name='l_hid2_drop')
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10, name='l_out')
    l_out_drop = GaussianDropoutLayer(l_out, prior_std=1e-3,  
            nonlinearity=lasagne.nonlinearities.softmax, name='l_out_drop')
    return l_out_drop

def build_cnn(input_var=None, masks=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var, name='l_in')
    conv1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(3, 3), name='conv1',
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())
    conv2 = lasagne.layers.Conv2DLayer(
            conv1, num_filters=32, filter_size=(3, 3), name='conv2',
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())
    pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride=2)
    conv3 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3), name='conv3',
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())
    conv4 = lasagne.layers.Conv2DLayer(
            conv3, num_filters=32, filter_size=(3, 3), name='conv4',
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())
    pool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(3, 3), stride=2)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool2, p=.5),
            num_units=500, name='l_hid1',
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid1, p=.5),
            num_units=500, name='l_hid2',
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid2, p=.5),
            num_units=10, name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def reloadModel(file_name, input_var=None, masks=None):
    file = open(file_name, 'r')
    data = cPickle.load(file)
    file.close()
    keys = data.keys()
    print keys
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var, name='l_in')
    conv1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=data['conv1.W'], b=data['conv1.b'], name='conv1')
    conv2 = lasagne.layers.Conv2DLayer(
            conv1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=data['conv2.W'], b=data['conv2.b'], name='conv2')
    pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride=2)
    conv3 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=data['conv3.W'], b=data['conv3.b'], name='conv3')
    conv4 = lasagne.layers.Conv2DLayer(
            conv3, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=data['conv4.W'], b=data['conv4.b'], name='conv4')
    pool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(3, 3), stride=2)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool2, p=.5),
            num_units=500, name='l_hid1', W=data['l_hid1.W'], b=data['l_hid1.b'], 
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid1, p=.5),
            num_units=500, name='l_hid2', W=data['l_hid2.W'], b=data['l_hid2.b'], 
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid2, p=.5),
            num_units=10, name='l_out', W=data['l_out.W'], b=data['l_out.b'])
    l_soft= SoftermaxNonlinearity(l_out, temp=1.)
    return (l_soft, l_out)


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
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model == 'cnn':
        network = build_cnn(input_var)
    elif model == 'reload':
        network = reloadModel(file_name, input_var=input_var)
    else:
        print("Unrecognized model type %r." % model)

    

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    batch_size = 250
    margin_lr = 25
    prediction = lasagne.layers.get_output(network, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.sum()
    # We could add some weight decay as well here, see lasagne.regularization.    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    
    for layer in lasagne.layers.get_all_layers(network):
        if hasattr(layer, 'W'):
            layer.W = lasagne.updates.norm_constraint(layer.W, L2Radius)
            #layer.W = L2BallConstraint(layer.W, L2Radius)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate = T.fscalar('learning_rate')
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
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

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
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
            val_err += err
            val_acc += acc
            val_batches += 1

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

def run_once(model='mlp', file_name=None, save_name='./models/model.npz'):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    dataset_size = X_train.shape[0]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model == 'reload':
        network = reloadModel(file_name, input_var=input_var)
    else:
        print("Unrecognized model type %r." % model)
    
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # After training, we compute and print the test error:
    test_err = 0.
    test_acc = 0.
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
    return test_acc / test_batches
    
def get_learning_rate(epoch, margin, base):
    return base*margin/np.maximum(epoch,margin)

def save_model(model, file_name):
    '''Save the model parameters'''
    print('Saving model..')
    params = {}
    for param in lasagne.layers.get_all_params(model):
        params[str(param)] = param.get_value()
    
    file = open(file_name, 'w')
    cPickle.dump(params, file, cPickle.HIGHEST_PROTOCOL)
    file.close()
                
class FullGaussianLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, nonlinearity,
                 M=None, R=None, **kwargs):
        super(FullGaussianLayer, self).__init__(incoming, **kwargs)
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

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        b = T.ones_like(input[:,0]).dimshuffle(0,'x')
        X = T.concatenate([input,b],axis=1)
        smrg = MRG_RandomStreams()
        self.E = smrg.normal(size=self.M.shape)
        self.W = self.M + self.S*self.E
        H = T.dot(X,self.W)
        # Nonlinearity
        return self.nonlinearity(H)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class GaussianDropoutLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nonlinearity,
                 R=None, prior_std=None, **kwargs):
        super(GaussianDropoutLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(incoming.output_shape[1:]))
        if prior_std == None:
            self.prior_std = 0.5
        else:
            self.prior_std = prior_std
        if R is None:
            R = lasagne.init.Constant(np.log(np.exp(self.prior_std)-1.))
        self.R = self.add_param(R, (num_inputs,), name='R')
        self.S = T.log(1. + T.exp(self.R)).dimshuffle('x',0)
        self.nonlinearity = nonlinearity
        self.layer_type = 'GaussianDropoutLayer'

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        # Nonlinearity
        smrg = MRG_RandomStreams()
        self.E = smrg.normal(size=input.shape)
        self.alpha = 1.+self.E*self.S
        return self.nonlinearity(input*self.alpha)

class SoftermaxNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, temp=1, **kwargs):
        super(SoftermaxNonlinearity, self).__init__(incoming, **kwargs)
        self.temp = temp

    def get_output_for(self, input, **kwargs):
        input = input/self.temp
        return T.exp(input)/T.sum(T.exp(input), axis=1).dimshuffle(0,'x')

def npSofterMax(logits, temp):
    data = np.exp(logits/temp)
    return data/np.sum(data, axis=1)[:,np.newaxis]

def GaussianRegulariser(W, E, M, S, Sp, prior = 'Gaussian'):
    '''Return cost of W'''
    if prior == 'Gaussian':
        return 0.5*(-T.sum(E**2) + T.sum(W**2)/(Sp**2)) - T.sum(T.log(S))
    elif prior == 'Laplace':
        return -0.5*T.sum(E**2) + T.sum(T.abs_(W))/(Sp*T.sqrt(2.)) - T.sum(T.log(S))
    else:
        print('Invalid regulariser')
        sys.exit(1)

def GaussianDropoutRegulariser(E, S, Sp):
    '''Return the cost of the Half Gaussian regularised layer'''
    return 0.5*(-T.sum(E**2) + T.sum((S*E)**2)/(Sp**2)) - T.sum(T.log(S))

def L2BallConstraint(tensor_var, target_norm, norm_axes=None, epsilon=1e-7):
    ndim = tensor_var.ndim
    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )
    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    constrained_output = \
        (tensor_var * (target_norm / (dtype(epsilon) + norms)))

    return constrained_output

def KL(input):
    w1, w2, S1, S2, b1, b2, x = input
    m1 = T.dot(x,w1) + b1
    m2 = T.dot(x,w2) + b2
    s1 = (S1*m1)**2
    s2 = (S2*m2)**2
    
    KL_div = 0.5*(2*T.log(s2/s1) + (s1/s2)**2 + ((m1-m2)/s2)**2)
    return KL_div.sum()

def modelTransfer(file_name, save_name='./models/newmodel.npz',
                  deterministic=True, copy_temp=1, mode='me'):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    dataset_size = X_train.shape[0]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    temp = T.fscalar('temp')
    # Create transfer model (depending on first command line parameter)
    print('Reloading transfer model')
    if deterministic == True:
        y_train = copy_model_output(file_name).astype(theano.config.floatX)
        target_probs = T.fmatrix('target_probs')
    else:
        transfer = reloadModel(file_name, input_var=input_var, temp=temp)
        target_probs = lasagne.layers.get_output(transfer, deterministic=False)
    print('Building approximate model')
    approx = build_mlp(input_var=input_var, temp=temp)
    prediction = lasagne.layers.get_output(approx, deterministic=False)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    batch_size = 500
    margin_lr = 25
    num_epochs = 500
    base_lr = 3e-5
    
    if mode == 'me':
        loss = -T.sum(prediction*T.log(target_probs), axis=1)
        reg = T.sum(prediction*T.log(prediction), axis=1)
        loss = (loss + reg).sum()
    elif mode == 'hinton':
        loss = -T.sum(target_probs*T.log(prediction), axis=1)
        loss = loss.sum()
    # Create update expressions for training
    params = lasagne.layers.get_all_params(approx, trainable=True)
    learning_rate = T.fscalar('learning_rate')
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(approx, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    if deterministic == True:
        train_fn = theano.function([input_var, target_probs, learning_rate, temp],
            loss, updates=updates)
    else:
        train_fn = theano.function([input_var, learning_rate, temp],
            loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var, temp], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        learning_rate = get_learning_rate(epoch, margin_lr, base_lr)
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        i = 0
        temp = get_temperature(epoch, copy_temp, 50.)
        print temp
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            targets = npSofterMax(targets, temp)
            if deterministic == True:
                train_err += train_fn(inputs, targets,
                                      learning_rate=learning_rate, temp=temp)
            else:
                train_err += train_fn(inputs, temp=temp,
                                      learning_rate=learning_rate)
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
            err, acc = val_fn(inputs, targets, temp=1)
            val_err += err
            val_acc += acc
            val_batches += 1

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
        err, acc = val_fn(inputs, targets, temp=1)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    acc = test_acc / test_batches * 100
    print("  test accuracy:\t\t{:.2f} %".format(acc))
    print('Complete')
    return acc

def get_temperature(epoch, copy_temp, beta):
    return 1. #(1. + np.exp(-(1.*epoch)/beta)*(copy_temp - 1.)).astype(theano.config.floatX)


def copy_model_output(file_name):
     # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    dataset_size = X_train.shape[0]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    # Create transfer model (depending on first command line parameter)
    print('Reloading transfer model')
    transfer, logits = reloadModel(file_name, input_var=input_var)
    target_probs = lasagne.layers.get_output(logits, deterministic=True)

    # Compile a function yielding outputs
    fnc = theano.function([input_var], target_probs)
    # Finally, launch the training loop.
    print("Starting copying...")
    # And a full pass over the data:
    batch_size=500
    labels = []
    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
        inputs, targets = batch
        labels.append(fnc(inputs))
    labels = np.vstack(labels)
    return labels

def scan_temperatures(save_name):
    temp = np.linspace(1.0, 20.0, num=8).astype(theano.config.floatX)
    accuracy = np.zeros((temp.shape[0],3))
    for i, t in enumerate(temp):
        accuracy[i,0] = t
        print('Me: Temperature %f' % (t,))
        accuracy[i,1] = modelTransfer('./models/mnistcnn.npz', deterministic=True,
                                      copy_temp=t, mode='me')
        print('Geoff: Temperature %f' % (t,))
        accuracy[i,2] = modelTransfer('./models/mnistcnn.npz', deterministic=True,
                                      copy_temp=t, mode='hinton')
    print accuracy
    np.save(save_name, accuracy)

def plot_temperatures(save_name):
    accuracy = np.load(save_name)
    fig = plt.figure()
    plt.plot(accuracy[:,0], accuracy[:,1], 'b')
    plt.plot(accuracy[:,0], accuracy[:,2], 'r')
    #plt.ylim(80,100)
    plt.show()

def analyse_soft_targets(file_name, save_name):
    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')
    temp = T.fscalar('temp')
    print('Reloading model')
    temps = np.linspace(1.,20.,19)
    mspread = np.zeros(temps.shape)
    vspread = np.zeros(temps.shape)
    mentropy = np.zeros(temps.shape)
    ventropy = np.zeros(temps.shape)
    mvariance = np.zeros(temps.shape)
    vvariance = np.zeros(temps.shape)
    for i, temp in enumerate(temps):
        y_train = copy_model_output(file_name, copy_temp=temp)
        spread = np.amax(y_train, axis=1) - np.amin(y_train, axis=1)
        mspread[i] = np.mean(spread)
        vspread[i] = np.var(spread)
        entropy = np.sum(y_train*np.log(y_train), axis=1)
        mentropy[i] = np.mean(entropy)
        ventropy[i] = np.var(entropy)
        variance = np.var(y_train, axis=1)
        mvariance[i] = np.mean(variance)
        vvariance[i] = np.var(variance)
    np.savez(save_name, mspread=mspread, vspread=vspread,
             mentropy=mentropy, ventropy=ventropy,
             mvariance=mvariance, vvariance=vvariance)

def plot_soft_targets(save_name):
    data = np.load(save_name)
    fig = plt.figure()
    plt.semilogy(np.linspace(1.,20.,19),data['mspread'], 'b')
    plt.plot(np.linspace(1.,20.,19),data['vspread'], 'b--')
    plt.plot(np.linspace(1.,20.,19),-data['mentropy'], 'r')
    plt.plot(np.linspace(1.,20.,19),data['ventropy'], 'r--')
    plt.plot(np.linspace(1.,20.,19),data['mvariance'], 'g')
    plt.plot(np.linspace(1.,20.,19),data['vvariance'], 'g--')
    plt.show()

def display_soft_targets(file_name):
    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')
    temp = 1
    y_train = copy_model_output(file_name, copy_temp=temp)
    for i in np.random.choice(y_train.shape[0], 10):
        fig = plt.figure()
        plt.bar(np.arange(10), np.log(y_train[i,:]))
        plt.show()
    

if __name__ == '__main__':
    main(model='cnn', save_name='./models/mnistcnn.npz', dataset='MNIST',
         num_epochs=100, L2Radius=3.87, base_lr=1e-4)
    #run_once(model='reload', file_name='./models/mnistcnn.npz')
    #modelTransfer('./models/mnistcnn.npz', deterministic=True, copy_temp=20,
    #              mode='hinton')
    #scan_temperatures('./models/accuracies_lr.npy')
    #plot_temperatures('./models/accuracies.npy')
    #analyse_soft_targets('./models/mnistcnn.npz', './models/soft_targets.npz')
    #plot_soft_targets('./models/soft_targets.npz')
    #display_soft_targets('./models/mnistcnn.npz')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
