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


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.

def build_mlp(input_var=None, masks=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var, name='l_in')
    conv1 = GaussianConv2DLayer(l_in, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv1')
    conv2 = GaussianConv2DLayer(conv1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv2')
    pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride=2)
    l_hid1 = FullGaussianLayer(pool1, num_units=500, name='l_hid1',
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = FullGaussianLayer(
            l_hid1, num_units=500, name='l_hid2',
            nonlinearity=lasagne.nonlinearities.rectify)
    l_out = FullGaussianLayer(
            l_hid2, num_units=10, name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def reloadModel(file_name, input_var=None, masks=None):
    file = open(file_name, 'r')
    data = cPickle.load(file)
    file.close()
    if masks is None:
        masks = {}
        masks['l_hid1'] = None
        masks['l_hid2'] = None
        masks['l_out'] = None
    
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var, name='l_in')
    l_hid1 = FullGaussianLayer(
            l_in, num_units=800, name='l_hid1', M=data['Ml_hid1'],
            R=data['Rl_hid1'], mask=masks['l_hid1'],
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = FullGaussianLayer(
            l_hid1, num_units=800, name='l_hid2', M=data['Ml_hid2'],
            R=data['Rl_hid2'], mask=masks['l_hid2'], 
            nonlinearity=lasagne.nonlinearities.rectify)
    l_out = FullGaussianLayer(
            l_hid2, num_units=10, name='l_out', M=data['Ml_out'],
            R=data['Rl_out'], mask=masks['l_out'],
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

def main(model='mlp', num_epochs=500, file_name=None, proportion=0.,
         save_name='./models/model.npz'):
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
    elif model == 'prune':
        network = prune(file_name, proportion, scheme='KL', input_var=input_var)
    else:
        print("Unrecognized model type %r." % model)

    

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prior_std = np.sqrt(1e0)
    batch_size = 500
    base_lr = 0.00001
    margin_lr = 50
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.sum()
    # We could add some weight decay as well here, see lasagne.regularization.
    reg = 0
    
    for layer in lasagne.layers.get_all_layers(network):
        if hasattr(layer, 'layer_type'):
            if layer.layer_type == 'GaussianLayer':
                reg += GaussianRegulariser(layer.W, layer.E,
                                          layer.M, layer.S,
                                          prior_std, prior='Gaussian')
    loss = loss + reg/T.ceil(dataset_size/batch_size)
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
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
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets, learning_rate=learning_rate)
            train_batches += 1

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

def run_once(model='mlp', file_name=None, proportion=0., scheme='KL',
         save_name='./models/model.npz'):
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
    elif model == 'prune':
        network = prune(file_name, proportion, scheme=scheme, input_var=input_var)
    else:
        print("Unrecognized model type %r." % model)
    
    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

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
    return test_acc / test_batches
    
def get_learning_rate(epoch, margin, base):
    return base*margin/np.maximum(epoch,margin)

def save_model(model, file_name):
    '''Save the model parameters'''
    print('Saving model..')
    params = {}
    for layer in lasagne.layers.get_all_layers(model):
        if hasattr(layer, 'layer_type'):
            if layer.layer_type == 'GaussianLayer':
                M = layer.M.get_value()
                R = layer.R.get_value()
                params['M' + layer.name] = M
                params['R' + layer.name] = R
    file = open(file_name, 'w')
    cPickle.dump(params, file, cPickle.HIGHEST_PROTOCOL)
    file.close()
                

class FullGaussianLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, nonlinearity,
                 M=None, R=None, mask=None, **kwargs):
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
        if mask != None:
            self.mask = mask

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        b = T.ones_like(input[:,0]).dimshuffle(0,'x')
        X = T.concatenate([input,b],axis=1)
        smrg = MRG_RandomStreams()
        self.E = smrg.normal(size=self.M.shape)
        self.W = self.M + self.S*self.E
        if hasattr(self, 'mask'):
            H = T.dot(X,self.W*self.mask)
        else:
            H = T.dot(X,self.W)
        # Nonlinearity
        return self.nonlinearity(H)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class FullLaplaceLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, nonlinearity,
                 M=None, R=None, **kwargs):
        super(FullLaplaceLayer, self).__init__(incoming, **kwargs)
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
        self.E = LaplaceRNG(shape=self.M.shape)
        self.W = self.M + self.S*self.E
        H = T.dot(X,self.W)
        # Nonlinearity
        return self.nonlinearity(H)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class GaussianConv2DLayer(lasagne.layers.Layer):
    '''Gaussian Convolutional Layer'''
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 border_mode="valid", untie_biases=False, M=None, R=None,
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(GaussianConv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2)
        self.stride = lasagne.utils.as_tuple(stride, 2)
        self.border_mode = border_mode
        self.untie_biases = untie_biases
        self.convolution = convolution

        if self.border_mode not in ['valid', 'full', 'same']:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)
        
        if M is None:
            M = lasagne.init.Constant(0.0)
        if R is None:
            r = np.log(np.exp(np.sqrt(1./filter_size[1]))-1.)
            R = lasagne.init.Constant(r)
        self.M = self.add_param(M, self.get_M_shape(), name='M')
        self.R = self.add_param(R, self.get_M_shape(), name='R')
        self.S = T.log(1. + T.exp(self.R))
        self.layer_type = 'GaussianConv2DLayer'
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_M_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        output_rows = lasagne.layers.conv.conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         self.border_mode)

        output_columns = lasagne.layers.conv.conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            self.border_mode)

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape
        filter_shape = self.get_M_shape()
        smrg = MRG_RandomStreams()
        self.E = smrg.normal(size=self.M.shape)
        self.W = self.M + self.S*self.E
        if self.border_mode in ['valid', 'full']:
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode)
        elif self.border_mode == 'same':
            if self.stride != (1, 1):
                raise NotImplementedError("Strided convolution with "
                                          "border_mode 'same' is not "
                                          "supported by this layer yet.")

            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

def LaplaceRNG(shape):
    smrg = MRG_RandomStreams()
    U = smrg.uniform(size=shape, low=-0.499, high=0.499)
    return -T.sgn(U)*T.log(1.-2.*T.abs_(U))/T.sqrt(2.)    

def GaussianRegulariser(W, E, M, S, Sp, prior = 'Gaussian'):
    '''Return cost of W'''
    if prior == 'Gaussian':
        return 0.5*(-T.sum(E**2) + T.sum(W**2)/(Sp**2)) - T.sum(T.log(S))
    elif prior == 'Laplace':
        return -0.5*T.sum(E**2) + T.sum(T.abs_(W))/(Sp*T.sqrt(2.)) - T.sum(T.log(S))
    else:
        print('Invalid regulariser')
        sys.exit(1)

def LaplaceRegulariser(W, E, M, S, Sp, prior='Laplace'):
    '''Regularise according to Laplace prior'''
    if prior == 'Laplace':
        return (T.sum(-T.abs_(E)) + T.sum(T.abs_(W))/Sp)*T.sqrt(2.) - T.sum(T.log(S))

def cumhist(SNR, nbins):
        '''Return normalised cumulative histogram of SNR'''
        SNR = np.hstack([SNR[snr].flatten() for snr in SNR])
        # Histogram of SNRs
        hist, bin_edges = np.histogram(SNR, bins=nbins)
        hist = np.cumsum(hist)
        hist = hist/(hist[-1]*1.)
        return (hist, bin_edges)

def histogram(model, scheme='KL'):
    SNR = {}
    for layer in lasagne.layers.get_all_layers(model):
        if hasattr(layer, 'layer_type'):
            if layer.layer_type == 'GaussianLayer':
                M = layer.M.get_value()[:-1,:]
                R = layer.R.get_value()[:-1,:]
                S = np.log(1. + np.exp(R))
                if scheme == 'KL':
                    snr = (M/S)**2 + 2.*np.log(S)
                    snr_min = np.amin(snr)
                    SNR[layer.name] = np.log(snr - snr_min + 1e-6)
                elif scheme == 'SNR':
                    snr = (M/S)**2 
                    snr_min = np.amin(snr)
                    SNR[layer.name] = np.log(snr - snr_min + 1e-6)
                elif scheme == 'lowest':
                    snr = np.abs(M)
                    snr_min = np.amin(snr)
                    SNR[layer.name] = np.log(snr - snr_min + 1e-6)
    hist, bin_edges = cumhist(SNR, 1000)
    bin_edges = bin_edges[1:]
    return (bin_edges, hist, SNR)
    
def prune(file_name, proportion, scheme='KL', input_var=None):
    '''Prune weights according to appropriate scheme'''
    model = reloadModel(file_name)
    bin_edges, hist, SNR = histogram(model, scheme=scheme)
    idx = (hist > proportion)
    cutoff = np.compress(idx, bin_edges)
    cutoff = np.amin(cutoff)
    masks = {}
    for snr in SNR:
        msk = (SNR[snr] > cutoff)
        newrow = np.ones((1, msk.shape[1]))
        msk = np.vstack((msk, newrow))
        masks[snr] = np.asarray(msk)
    return reloadModel(file_name, input_var=input_var, masks=masks)
    
def plotToPrune(model):
    '''Plot the weight histograms'''
    schemes = ['KL', 'SNR', 'lowest']
    fig = plt.figure()
    for scheme in schemes:
        bin_edges, hist, _ = histogram(model, scheme=scheme)
        plt.plot(bin_edges, hist)
    plt.show()

def plottests(num_steps):
    proportion = 1.-np.logspace(-3, -1, num_steps)
    acc = np.zeros((proportion.shape[0],4))
    for i, prop in enumerate(proportion):
        print prop
        acc[i,0] = prop
        acc[i,1] = run_once(model='prune', file_name='./models/modelG0.npz',
                       proportion=prop, scheme='KL')
        acc[i,2] = run_once(model='prune', file_name='./models/modelG0.npz',
                       proportion=prop, scheme='SNR')
        acc[i,3] = run_once(model='prune', file_name='./models/modelG0.npz',
                       proportion=prop, scheme='lowest')
    fig = plt.figure()
    plt.semilogx(1-acc[:,0], acc[:,1:])
    plt.show()
    np.save('./models/PG0.npy', acc)

if __name__ == '__main__':
    main(save_name='./models/modelCG0.npz')
    #run_once(model='prune', file_name='./models/modelG0.npz', proportion=0.99,
    #        scheme='KL')
    #plottests(25)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
