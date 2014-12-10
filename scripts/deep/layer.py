'''
The layer class creates a symbolic object storing the parameters of a
single layer of a layered neural network (MLP, Autoencoder, DBM, etc..)
with a defined connectivity. The specific choice of nonlinearity and
input domain are explicitly excluded - those have to be defined outside
in a separate script controlling overall topology and functionality.
'''

import numpy as np

import theano
import theano.tensor as T


class Layer(object):
    '''Layer class
    
    The layer class creates a symbolic object storing the parameters of a
    single layer of a layered neural network (MLP, Autoencoder, DBM, etc..)
    with a defined connectivity. The specific choice of nonlinearity and
    input domain are explicitly excluded - those have to be defined outside
    in a separate script controlling overall topology and functionality.
    '''
    def __init__(
        self,
        n_in=784,
        n_out=500,
        W=None,
        b=None,
        b_in=None,
        mask=None
        ):
        
        '''
        The layer class provides no initialisation. That is left to a separate
        script. The reason for this is to provide absolute separation between
        various aspects of the training and usage
        
        :type n_in: int
        :param n_in: number of inputs
        
        :type n_out: int
        :param n_out: number of outputs
        
        :type W: theano.tensor.TensorType
        :param W: connectivity matrix
        
        :type b: theano.tensor.TensorType
        :param b: output layer biases
        
        :type mask: theano.tensor.TensorType
        :param mask: mask specifiying specialist connectivities
        '''
        
        if b_in is not None:
            # b_in is an optional bias applied to the input of the first layer
            # only. If used in hidden layers it will create big problems, but that
            # option is left for the user (you never know, certain doubly-biased
            # hidden neurons may be of particular esoteric interest).

            b_in = theano.shared(value=b_in, name='b_in', borrow=True)
            self.b_in = b_in
            print 'Input bias constructed'
        
        if not b:
            # b is initialised as all zeros. This is common. If the user wants
            # to coax sparsity, it is possible to subtract 4 (or there abouts)
            # from the the biases.
            
            initial_b = np.zeros(shape=(n_out), dtype=theano.config.floatX)
            b = theano.shared(value=initial_b, name='b', borrow=True)
            print 'Bias constructed'
            

        if not W:
            # W is initialised as all zeros. This forces the user to include an
            # initialisation routine over all layers in order to break symmetry.
            
            initial_W = np.zeros(shape=(n_in, n_out), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
            print 'Weight constructed'
        
        self.W = W
        self.b = b
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        