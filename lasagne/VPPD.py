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

from lasagne import utils
from collections import OrderedDict
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
            l_in, num_units=400,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_hid2 = lasagne.layers.DenseLayer(
            l_in, num_units=400,
            W=lasagne.init.GlorotUniform(), name='l_hid2')
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=10,
            W=lasagne.init.GlorotUniform(), name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

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

def main(num_epochs=100, file_name=None, proportion=0.,
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
    network = build_mlp(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    batch_size = 100
    margin_lr = 25
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate = T.fscalar('learning_rate')
    # The prior
    log_prior = 0.
    for param in params:
        if param.name[-1] == 'W':
            print('Prior W')
            log_prior += -0.1*T.sum(param**2)
        elif param.name[-1] == 'b':
            print('Prior b')
            log_prior += -0.1*T.sum(param**2) 
    updates = SGLD(loss, params, learning_rate, log_prior, N=50000)
    mean_loss = loss.mean()
    #updates = nesterov_momentum(loss, params, learning_rate=learning_rate,
    #                            momentum=0.9)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, learning_rate],
        mean_loss, updates=updates)

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

def main2(num_epochs=100, file_name=None, save_name='./models/model.npz',
          dataset='MNIST', L2Radius=3.87, base_lr=0.0003):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset)
    dataset_size = X_train.shape[0]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learning_rate = T.fscalar('learning_rate')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    teacher = build_mlp(input_var)
    student = build_mlp(input_var)
    # Hyperparameters
    batch_size = 500
    margin_lr = 25
    burn_in = 100
    # Networks
    t_pred = lasagne.layers.get_output(teacher)
    s_pred = lasagne.layers.get_output(student)
    # Loss functions
    t_loss = lasagne.objectives.categorical_crossentropy(t_pred, target_var)
    t_loss = t_loss.mean()
    t_params = lasagne.layers.get_all_params(teacher, trainable=True)
    # Sample the teacher network weight posterior
    # The Gaussian weight/bias prior
    log_prior = 0.
    for param in t_params:
        if param.name[-1] == 'W':
            print('Prior W')
            log_prior += -0.1*T.sum(param**2)
        elif param.name[-1] == 'b':
            print('Prior b')
            log_prior += -0.1*T.sum(param**2) 
    t_updates = SGLD(t_loss, t_params, learning_rate, log_prior, N=50000)
    # SGD on the student network parameters
    s_loss = T.mean(s_pred*(T.log(s_pred)-T.log(t_pred)))
    s_params = lasagne.layers.get_all_params(teacher, trainable=True)
    s_updates = nesterov_momentum(s_loss, s_params, learning_rate=learning_rate,
                                  momentum=0.9)
    # Compile functions
    t_acc = T.mean(T.eq(T.argmax(t_pred, axis=1), target_var),
                      dtype=theano.config.floatX)
    t_fn = theano.function([input_var, target_var, learning_rate],
        [t_loss, t_acc], updates=t_updates)
    s_fn = theano.function([input_var, learning_rate], updates=s_updates)

    # Compile a second function computing the validation loss and accuracy:
    s_tar_loss = lasagne.objectives.categorical_crossentropy(s_pred, target_var)
    s_tar_acc = T.mean(T.eq(T.argmax(s_pred, axis=1), target_var),
                      dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var],
        [s_tar_loss.mean(), s_tar_acc])

    # Finally, launch the training loop.
    print("Burning in")
    for epoch in range(burn_in):
        learning_rate = get_learning_rate(epoch, margin_lr, base_lr)
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        t_err = 0
        t_accu = 0
        t_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            # Sample weights from teacher
            err, acc = t_fn(inputs, targets, learning_rate=learning_rate)
            t_err += err
            t_accu += acc
            t_batches += 1
            
        # Then we print the results for this epoch:
        print("Burn {} of {} took {:.3f}s".format(
            epoch + 1, burn_in, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(t_err / t_batches))
        print("  training acc:\t\t{:.6f}".format(t_accu / t_batches))
        
    # We iterate over epochs:
    print("Knowledge transfer")
    for epoch in range(burn_in+num_epochs):
        learning_rate = get_learning_rate(epoch, margin_lr, base_lr)
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        t_err = 0
        t_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            # Sample weights from teacher
            err, _ = t_fn(inputs, targets, learning_rate=learning_rate)
            t_err += err
            # Train student
            s_fn(inputs, learning_rate)
            t_batches += 1

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
        print("  training loss:\t\t{:.6f}".format(t_err / t_batches))
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

class SoftermaxNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, temp=1, **kwargs):
        super(SoftermaxNonlinearity, self).__init__(incoming, **kwargs)
        self.temp = temp

    def get_output_for(self, input, **kwargs):
        input = input/self.temp
        return T.exp(input)/T.sum(T.exp(input), axis=1).dimshuffle(0,'x')

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
        updates[param] = param - learning_rate * grad
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

def SGLD(loss, params, learning_rate, log_prior, N):
    """Apply the SGLD MCMC sampler"""
    g_lik = N*get_or_compute_grads(-loss, params)
    g_prior = get_or_compute_grads(log_prior, params)
    smrg = MRG_RandomStreams()
    updates = OrderedDict()
    for param, gl, gp in zip(params, g_lik, g_prior):
        eta = T.sqrt(learning_rate)*smrg.normal(size=param.shape)
        delta = 0.5*learning_rate*(gl + gp) #+ eta
        updates[param] = param + delta
    return updates
    


if __name__ == '__main__':
    main2(save_name='./models/mnistVPPD.npz', dataset='MNIST',
         num_epochs=500, L2Radius=3.87, base_lr=1e-3)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
