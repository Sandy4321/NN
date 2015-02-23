'''
The layer class creates a symbolic object storing the parameters of a
single layer of a layered neural network (MLP, Autoencoder, DBM, etc..)
with a defined connectivity. The specific choice of nonlinearity and
input domain are explicitly excluded - those have to be defined outside
in a separate script controlling overall topology and functionality.

We avoid sanity check for now, which will only hinder getting a system
up and running fast.

@author: dew
@date: 8 Jan 2013
'''

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tnet
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import matplotlib.pyplot as plt



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
        input=None,
        layer_type='AE',
        nonlinearity='logistic',
        h_reg='xent',
        W_reg='L2',
        np_rng=None,
        theano_rng=None,
        W=None,
        b=None,
        b2=None
        ):
        
        '''       
        The layer class provides no initialisation. That is left to a separate
        script. The reason for this is to provide absolute separation between
        various aspects of the training and usage
        
        :type v_n: int
        :param v_n: number of visible units
        
        :type h_n: int
        :param h_n: number of hidden units
        
        :type input: theano.tensor.TensorType
        :param input: input to layer
        
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
        
        if (layer_type == 'AE') or (layer_type == 'DAE'):
            initial_b2 = np.zeros(shape=(v_n), dtype=theano.config.floatX)
            b2 = theano.shared(value=initial_b2, name='b2', borrow=True)
            self.b2 = b2
            self.W_prime = self.W.T

            # create a numpy and Theano random generator that give symbolic random values
            if np_rng is None:
                self.np_rng = np.random.RandomState(123)
            else:
                self.np_rng = np_rng
        
            if theano_rng is None:
                self.theano_rng = RandomStreams(self.np_rng.randint(2 ** 30))
            else:
                self.theano_rng = theano_rng
            
        self.layer_type     = layer_type
        self.nonlinearity   = nonlinearity
        self.h_reg          = h_reg
        self.W_reg          = W_reg
        self.h_n            = h_n
        self.v_n            = v_n
        
        if b2 is not None:
            self.params = [self.W, self.b, self.b2]
        else:
            self.params = [self.W, self.b]
            
        # declare input and output
        if input == None:
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        self.output = self.get_enc(self.x)
        
        
    def init_weights(self, initialisation_regime, nonlinearity):
        
        if initialisation_regime == 'None':
            pass
        elif initialisation_regime == 'Glorot':
            W_shape = self.W.get_value(borrow=True, return_internal_type=True).shape

            if (nonlinearity == 'tanh') or (nonlinearity == 'tanh_linear'):
                r = 0.1*np.sqrt(6.0/(np.sum(W_shape)))
            elif (nonlinearity == 'logistic') or (nonlinearity == 'logistic_linear'):
                r = 4.0*np.sqrt(6.0/(np.sum(W_shape)))
            elif (nonlinearity == 'linear') or (nonlinearity == 'softplus'):
                r = 0.1        # similarish to a Gaussian with variance 0.01
            else:
                print 'Invalid nonlinearity to initialise'
                exit(1)
            
            np_rng = 2*r*(np.random.random_sample(size=W_shape).astype(dtype=theano.config.floatX)-0.5)
            self.W.set_value(np_rng)
        else:
            print('Invalid initalisation regime')
            sys.exit(1)
        
    
    def init_random_numbers(self, mode, shape):
        """
        Despite all the lovely things about theano the random number generation
        is one of its frustrating sticking points. Not only are they harder to use
        but they are also very buggy. It appears that we need to create a numpy
        rng object and pass that through to a theano shared variable, which will
        presumably be shipped to the GPU during graph construction. The various
        rng objects supplied by thenano run at speeds differing by orders of magnitude
        so the solution found here is really just an empirical hack to speed things.
        
        We create dedicated shared RNGs for the weight matrices of each layer. These
        RNGs have a set shape and input distribution, which can only be defined once.
        
        :type mode: string
        :param mode: RNG type ('bernoulli', 'salt_and_pepper', 'gaussian')
        
        :type shape: int tuple
        :param shape: shape of RNG
        
        RETURNS
        :type: int
        :param: index of first RNG (assume we created [index] through to [-1])
        """
        if not hasattr(self, 'rng'):
            self.rng = ()
            
        if (mode == 'bernoulli'):
            rng     = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            self.rng += (rng,)
            indices = 1
        elif (mode == 'salt_and_pepper'):
            rnga    = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            rngb    = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            self.rng += (rnga,rngb)
            indicies = 2
        elif mode == 'gaussian':
            rng     = theano.shared(np.asarray(self.np_rng.randn(shape)), \
                                     theano.config.floatX).astype(theano.config.floatX)
            self.rng += (rng,)
            indicies = 1
        else:
            print('Invalid noise type for initialisation')
            sys.exit(1)
        
        return len(self.rng) - indicies
    
    
    def destroy_random_numbers(self):
        if hasattr(self, 'rng'):
            del self.rng
    
    
    
    def get_corrupt(self, input, corruption_level):
        """
        This corresponds to noise injected at the input to the layer
        
        :type input: theano.config.floatX
        :param input: the matrix of inputs to corrupt
        
        :type corruption_level: float in [0,1]
        :param corruption_level: discrete corruption probability/continuous noise variance
        """
        if self.noise_type == 'bernoulli':
            return  (self.rng[0]>corruption_level*self.get_cm()) * input
        elif self.noise_type == 'gaussian':
            return self.rng[0]*T.sqrt(corruption_level*self.get_cm()) + input
        elif self.noise_type == 'salt_and_pepper':
            a = (self.rng[0]>(corruption_level*self.get_cm()))*1
            b = (self.rng[1]>0.5)*1
            c = T.eq(a,0) * b
            return (input*a) + c
        else:
            print('Invalid noise type')
            sys.exit(1)
            
            
    
    def get_cm(self):
        '''
        Get corruption multiplier
        
        :type iteration: int
        :param iteration: used for annealing noise
        '''
        print('Corruption scheme: ' + self.corruption_scheme)
        if self.corruption_scheme == 'anneal':
            multiplier = 1.0*self.corruption_tau/np.amax((self.iteration, self.corruption_tau))
        elif self.corruption_scheme == 'random':
            multiplier = T.cast(np.random.random_sample(), dtype=theano.config.floatX)
        elif self.corruption_scheme == 'flat':
            multiplier = 1.0
        else:
            print('Invalid pretraining corruption scheme')

        return multiplier
    
    
    
    def set_iteration(self, iteration):
        self.iteration = iteration
    
    
    def get_enc(self, visible):
        '''
        Computes the output of a hidden layer
        
        :type visible: theano.config.floatX
        :param visible: the input to the layer
        '''
        if self.nonlinearity == 'logistic':
            output = Tnet.sigmoid(T.dot(visible,self.W) + self.b)
        elif self.nonlinearity == 'linear':
            output = T.dot(visible,self.W) + self.b
        elif self.nonlinearity == 'logistic_linear':
            output = Tnet.sigmoid(T.dot(visible,self.W) + self.b)
        elif self.nonlinearity == 'softplus':
            output = Tnet.softplus(T.dot(visible,self.W) + self.b)
        elif self.nonlinearity == 'tanh':
            output = T.tanh(T.dot(visible,self.W) + self.b)
        elif self.nonlinearity == 'tanh_linear':
            output = T.tanh(T.dot(visible,self.W) + self.b)
        else:
            print('Invalid encoding nonlinearity')
            sys.exit(1)
        return output

    
    
    def get_dec(self, hidden):
        '''
        Computes the output of a layer given a hidden layer input
        
        :type hidden: theano.config.floatX
        :param hidden: the hidden layer input
        '''
        if self.nonlinearity == 'logistic':
            output = Tnet.sigmoid(T.dot(hidden,self.W_prime) + self.b2)
        elif self.nonlinearity == 'linear':
            output = T.dot(hidden,self.W_prime) + self.b2
        elif self.nonlinearity == 'logistic_linear':
            output = T.dot(hidden,self.W_prime) + self.b2
        elif self.nonlinearity == 'softplus':
            output = Tnet.softplus(T.dot(hidden,self.W_prime) + self.b2)
        elif self.nonlinearity == 'tanh':
            output = Tnet.sigmoid(T.dot(hidden,self.W_prime) + self.b2)
        elif self.nonlinearity == 'tanh_linear':
            output = T.dot(hidden,self.W_prime) + self.b2
        else:
            print('Invalid decoding nonlinearity')
            sys.exit(1)
        return output



    ### 1 LOAD PARAMETERS
    def load_pretrain_params(self,
                    loss_type,
                    optimisation_scheme,
                    layer_scheme,
                    n_train_batches,
                    batch_size              = 10,
                    pretrain_learning_rate  = 0.1,
                    pretrain_epochs         = 10,
                    initialisation_regime   = 'Glorot',
                    noise_type              = 'bernoulli',
                    corruption_level        = 0.1,
                    corruption_scheme       = 'flat',
                    corruption_tau          = 1):
        self.loss_type              = loss_type
        self.optimisation_scheme    = optimisation_scheme
        self.layer_scheme           = layer_scheme
        self.n_train_batches        = n_train_batches
        self.batch_size             = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.pretrain_epochs        = pretrain_epochs
        self.initialisation_regime  = initialisation_regime
        self.noise_type             = noise_type
        self.corruption_level       = corruption_level
        self.corruption_scheme      = corruption_scheme
        self.corruption_tau         = corruption_tau
        
        self.init_weights(initialisation_regime, self.nonlinearity)
        self.iteration              = 1.0


    ### 2 CONSTRUCT EXPRESSION GRAPH

    def get_cost_updates(self, learning_rate):
            '''
            This function is based on the theano example. It computes the costs and
            a parameter update for a single training step. We consider SGD for the
            time being.
            
            :type learning_rate: theano.config.floatX
            :param learning_rate: rate at which to perfrom gradient descent
            
            :type iteration: int
            :param iteration: used to calculate corruption noise
            '''
            if self.layer_scheme == 'DAE':
                self.init_random_numbers(mode=self.noise_type, shape=(self.batch_size,self.v_n))
                x_tilde = self.get_corrupt(self.x, self.corruption_level)
            else:
                x_tilde = self.x
            y = self.get_enc(x_tilde)
            z = self.get_dec(y)
            
            # Define loss
            if self.loss_type=="L2":
                L = 0.5*T.sum((z - self.x)**2, axis=1)
            elif self.loss_type=="AE_xent":
                L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            else:
                print 'Layer loss type not recognised'
                sys.exit(1)
            
            # need to define cost
            reg = 0
            cost = T.mean(L + reg)
            # need to find gradient
            gparams = T.grad(cost, self.params)
            
            # generate the list of updates via SGD learning rule
            updates = []
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - learning_rate * gparam))
            
            return cost, updates

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        