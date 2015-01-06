'''
The layer class creates a symbolic object storing the parameters of a
single layer of a layered neural network (MLP, Autoencoder, DBM, etc..)
with a defined connectivity. The specific choice of nonlinearity and
input domain are explicitly excluded - those have to be defined outside
in a separate script controlling overall topology and functionality.

We avoid sanity check for now, which will only hinder getting a system
up and running fast.
'''

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tnet


class Layer(object):
    '''Layer class
    
    The layer class creates a symbolic object storing the parameters of a
    single layer of a layered neural network (MLP, Autoencoder, DBM, etc..)
    with a defined connectivity. Note that for autoencoders we only define
    half of the machine and efficiently store the parameters of the decoder
    (which we require to be weight-tied) in the encoder layers.
    '''
    
    ### 1 LOAD PARAMETERS
    
    def __init__(
        self,
        v_n=784,
        h_n=500,
        layer_type='AE',
        nonlinearity='sigmoid',
        h_reg='xent',
        W_reg='L2',
        W=None,
        b=None,
        b2=None,
        mask=None
        ):
        
        '''       
        The layer class provides no initialisation. That is left to a separate
        script. The reason for this is to provide absolute separation between
        various aspects of the training and usage
        
        :type v_n: int
        :param v_n: number of visible units
        
        :type h_n: int
        :param h_n: number of hidden units
        
        :type layer_type: string
        :param layer_type: what kind of machine layer we desire to build
        
        :type nonlinearity: string
        :param nonlinearity: type of feedforward nonlinearity
        
        :type h_reg: string
        :param h_reg: type of regularisation on hidden units
        
        :type W_reg: string
        :param W_reg: type of regularisation on weights
        
        :type W: theano.tensor.TensorType
        :param W: connectivity matrix
        
        :type b: theano.tensor.TensorType
        :param b: output layer biases
        
        :type mask: theano.tensor.TensorType
        :param mask: mask specifiying specialist connectivities
        '''
        
        if b2 is not None:
            # b2 is an optional bias applied to the input of the first layer
            # or as the decoder layer bias in autoencoders. If used in hidden
            # layers it will create big problems, but that option is left for
            # the user (you never know, certain doubly-biased hidden neurons
            # may be of particular esoteric interest).

            b2 = theano.shared(value=b2, name='b2', borrow=True)
            self.b2 = b2
        
        
        if b is None:
            # b is initialised as all zeros. This is common. If the user wants
            # to coax sparsity, it is possible to subtract 4 (or there abouts)
            # from the the biases.
            
            initial_b = np.zeros(shape=(h_n), dtype=theano.config.floatX)
            b = theano.shared(value=initial_b, name='b', borrow=True)
            

        if W is None:
            # W is initialised as all zeros. This forces the user to include an
            # initialisation routine over all layers in order to break symmetry.
            
            initial_W = np.zeros(shape=(v_n, h_n), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        
        
        self.W = W
        self.b = b
        
        if layer_type == 'AE':
            initial_b2 = np.zeros(shape=(v_n), dtype=theano.config.floatX)
            b2 = theano.shared(value=initial_b2, name='b2', borrow=True)
            self.b2 = b2
            self.W_prime = self.W.T
            
        self.layer_type = layer_type
        self.nonlinearity = nonlinearity
        self.h_reg = h_reg
        self.W_reg = W_reg
        if b2 is not None:
            self.params = [self.W, self.b, self.b2]
        else:
            self.params = [self.W, self.b]
        
        
        # Define encoding function - we are using different kinds of sigmoid for
        # the moment, but will generalise to more functions later on.
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
            sys.exit(1)
        
        self.enc_fn = theano.function([x], encoding_function)
        
        # Define decoding function - we are using different kinds of sigmoid for
        # the moment, but will generalise to more functions later on. Note that this
        # is only really useful for AEs
        
        y = T.matrix('y',dtype=theano.config.floatX)
        if self.nonlinearity == 'sigmoid':
            decoding_function = Tnet.sigmoid(T.dot(y,self.W_prime) + self.b2)
        elif self.nonlinearity == 'ultra_fast_sigmoid':
            decoding_function = ultra_fast_sigmoid(T.dot(y,self.W_prime) + self.b2)
        elif self.nonlinearity == 'hard_sigmoid':
            decoding_function = hard_sigmoid(T.dot(y,self.W_prime) + self.b2)
        elif self.nonlinearity == 'linear':
            decoding_function = (T.dot(y,self.W_prime) + self.b2)
        else:
            print("Decoding nonlinearity not supported")
            sys.exit(1)
        
        self.enc_fn = theano.function([x], encoding_function)
        self.dec_fn = theano.function([x], decoding_function)
        
    
    def get_enc(self, input):
        ''' Computes the output of a layer '''
        return self.enc_fn(input)

    
    def get_dec(self, input):
        ''' Computes the output of a layer '''
        return self.dec_fn(input)


    def get_recon(self, input):
        ''' Computes AE reconstruction '''
        return self.dec_fn(self.enc_fn(input))


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        