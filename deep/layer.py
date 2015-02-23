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
import theano.tensor.nnet as Tnet


class Layer(object):
    '''Layer class
    
    The layer class creates a symbolic object storing the parameters of a
    single layer of a layered neural network (MLP, Autoencoder, DBM, etc..)
    with a defined connectivity. 
    '''
    def __init__(
        self,
        n_in=784,
        n_out=500,
        nonlinearity='sigmoid',
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
        
        if b is None:
            # b is initialised as all zeros. This is common. If the user wants
            # to coax sparsity, it is possible to subtract 4 (or there abouts)
            # from the the biases.
            
            initial_b = np.zeros(shape=(n_out), dtype=theano.config.floatX)
            b = theano.shared(value=initial_b, name='b', borrow=True)
            

        if W is None:
            # W is initialised as all zeros. This forces the user to include an
            # initialisation routine over all layers in order to break symmetry.
            
            initial_W = np.zeros(shape=(n_in, n_out), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        
        self.W = W
        self.b = b
        self.nonlinearity = nonlinearity
        
        # Define encoding function
        x = T.matrix('x',dtype=theano.config.floatX)
        if self.nonlinearity == 'sigmoid':
            encoding_function = Tnet.sigmoid(T.dot(x,self.W) + self.b)
        elif self.nonlinearity == 'ultra_fast_sigmoid':
            encoding_function = ultra_fast_sigmoid(T.dot(x,self.W) + self.b)
        elif self.nonlinearity == 'hard_sigmoid':
            encoding_function = hard_sigmoid(T.dot(x,self.W) + self.b)
        elif self.nonlinearity == 'linear':
            encoding_function = (T.dot(x,self.W) + self.b)
        else:
            print("Encoding nonlinearity not supported")
        
        newW = T.matrix(name='newW', dtype=theano.config.floatX)
        self.enc_fn = theano.function([x], encoding_function)
        self.enc_fn2 = theano.function([x, newW], encoding_function, givens=[(self.W, newW)])
        
    
    def get_output(self, input):
        ''' Computes the output of a layer '''
        return self.enc_fn(input)
    
    def get_output2(self, input):
        ''' Computes the output of a layer '''
        return self.enc_fn2(input, newW)
    

    
    
    
    



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        