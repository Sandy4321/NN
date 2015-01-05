'''
The net class defines the overall topology of a neural networks, be it
directed or undirected. This is a very flexible setup which should give
the user a high degree of manipulative ability over the various aspects
of neural net training.
'''

import numpy as np
import cPickle
import gzip

import theano
import theano.tensor as T

from layer import Layer
from initialisation import Initialisation
from train import Train

class NN(object):
    ''' NN class
    
    We can define a general network topology. At the moment masks are not
    supported and all hidden nodes MUST bind ONE input to ONE output
    '''
    
    def __init__(
        self,
        topology=(784, 500, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot',
        pparams=None,
        input_bias=False,
        input=None
        ):
        '''
        :type topology: tuple of ints
        :param topology: tuple defining NN topology
        
        :type nonlinearities: tuple of strings
        :param nonlinearities: tuple defining the nonlinearities of the network
        
        :type initialisation: string
        :param initialisation: the type of initalisation precedure for the NN parameters
        
        :type pparams: list of named tuples
        :param pparams: contains pretrained parameters from a previous network
        
        :type input_bias: boolean
        :param input_bias: flag to denote whether there is an input bias in the first layer
        
        '''
        
        
        # We are going to store the layers of the NN in a net object, where
        # net[0] is the input layer etc. This has callable symbolic parameters
        # e.g. net[x].W, net[x].b
        
        self.net = []   
        
        # Make sure that the specfied nonlinearities correspond to layers,
        # otherwise throw an error and exit
        assert len(nonlinearities) == len(topology) - 1
        
        # It may be the case that we wish to load a premade model, in which
        # case it is important to check that the parameters match up with
        # the stated topology and we also need to bypass parameter initialisation.
        # Note that a pretrained model may only be fully pretrained. Partial
        # pretraining is yet to be supported
        
        
        if pparams is None:
            W=None,
            b=None,
            b_in=None,
            mask=None
        else:
            if  hasattr(pparams[0], 'b_in'):
                assert input_bias == True, "Need to set 'input_bias=True'"
                assert pparams[0].b_in.shape[0] == pparams[0].W.shape[1], \
                    "b_in or W wrong shape"
                print('Input biases consistent')
                
            for i in np.arange(0,len(pparams)-1):
                assert pparams[i].W.shape[1] == pparams[i+1].W.shape[0], \
                    "W connectivity mismatch between layer %i and layer %i" % (i, i+1)
                assert pparams[i].b.shape[0] == pparams[i].W.shape[0], \
                    "b/W connectivity mismatch in layer %i" % i
                assert self.supported_nonlinearity(pparams[i].nonlinearity), \
                    "Unsupported nonlinearity"
                print('Layer %i parameters are consistent' % (i))
            

            
        '''
        Now we simply loop through the elements of 'topology' and create a
        network layer with appropriate bound nonlinearities and initialised
        weights/biases.
        '''
        
        if pparams is None:
            # Build layers
            for i in np.arange(1,len(topology)):
                if (input_bias == True) and (i == 1):
                    # Instantiate layer with inputs biases
                    lyr = Layer(topology[i-1],topology[i], b_in = topology[i-1])
                else:
                    #Instantiate hidden/output layers
                    lyr = Layer(topology[i-1],topology[i])
                self.net.append(lyr)
            
            # Initialise weights of each constructed layer
            init_lyr = Initialisation()
            for lyr in self.net:
                init_lyr.init_weights(lyr, command='Glorot', nonlinearity='sigmoid')
                print(lyr, 'built')
            
            del init_lyr
        else:
            # Build layers
            for i in np.arange(1,len(topology)):
                if (input_bias == True) and (i == 1):
                    # Instantiate layer with inputs biases
                    lyr = Layer(W=pparams[0].W, b=pparams[0].b, b_in=pparams[0].b_in)
                else:
                    #Instantiate hidden/output layers
                    lyr = Layer(W=pparams[i-1].W, b=pparams[i-1].b)
                self.net.append(lyr)
                print(lyr, "built")
        
        # It will turn out advantageous later to easily access these variables
        self.num_layers = len(topology) - 1
        self.topology = topology
        self.nonlinearities = nonlinearities
        self.intialisation = initialisation
        self.pparams = pparams
        self.input_bias = input_bias
        # Run this command to load defaults.
        self.pretrain_params()
    
    
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
    
    
    def load_data(self, dataset):
        '''Load the dataset, which must be in pickled form. Will want to add a
        database retrieval function later
        
        Disclaimer: Copied straight from Montreal deep learning tutorials
        
        :type dataset: string
        :param dataset: path to dataset directory
        '''
        
        print('Loading data')
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')
    
        self.test_set_x, self.test_set_y = shared_dataset(test_set)
        self.valid_set_x, self.valid_set_y = shared_dataset(valid_set)
        self.train_set_x, self.train_set_y = shared_dataset(train_set)
    


    
    def pretrain_params(
        self, 
        method='AE',
        loss='SE',
        regulariser=('L2'),
        optimiser='SDG',
        momentum='0.1',
        scheduler='ED'
        ):
        '''
        Load pretraining parameters into every layer in the network.
        
        For now we force the pretrainer to apply the same scheme to every layer
        but in future we expect the training scheme to be per layer, hence the
        training commands are stored layer-wise.
        '''
        
        for lyr in self.net:
            lyr.method = method
            lyr.loss = loss
            lyr.regulariser = regulariser
            lyr.optimiser = optimiser
            lyr.momentum = momentum
            lyr.scheduler = scheduler
        
        # Note that for the autoencoder setup we need to specify a few extra parameters
        # These are whether the AE in symmetric in the representation layer and thus,
        # the kinds of weight-tying we would like to use. Furthermore, which half of the
        # parameters we may wish to discard.
        #
        # AE = symmetric + weight tying
        # AE_ns = non_symmetric (no weight tying)
        
        if method == 'AE':
            # Check for topology symmetry
            for i in xrange(self.num_layers):
                print i
                assert self.topology[i] == self.topology[self.num_layers - i], \
                    'AE-autoencoders need to symmetric in the representations layer'
        
        print("Pretraining parameters loaded")
            
    
    def pretrain(self):
        '''
        Greedy layer-wise pretrain the network
        '''
        
        train = Train()
        
        for layer_number in xrange(len(self.net)):
            train.train_layer_build(self, layer_number)
            train.train_layer(self, layer_number)
        
        


    
    
    
    
    
    
    
    
    def supported_nonlinearity(self,nonlinearity):
        return nonlinearity == 'sigmoid'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    