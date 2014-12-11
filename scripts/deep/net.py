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

class Net(object):
    ''' Net class
    
    We can define a general network topology. At the moment masks are not
    supported and all hidden nodes MUST bind ONE input to ONE output
    '''
    
    def __init__(
        self,
        topology=(784, 500, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot',
        pparams=None,
        input_bias=False
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
        
        
        # We are going to store the layers of the NN in a layers object,
        # where layers[0] is the input layer, layers[1] the first hidden
        # layer etc. This has callable symbolic parameters e.g. layers[x].W
        self.layers = []   
        
        # Make sure that the specfied nonlinearities correspond to layers,
        # otherwise throw and error and exit
        assert len(nonlinearities) == len(topology) - 1
        self.nonlinearities = nonlinearities
        
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
                assert pparams[0].b_in.shape == (topology[0],), \
                    "b_in wrong shape"
                assert input_bias == True, "Need to set 'input_bias=True'"
                print('Input biases consistent')
                
            for i in np.arange(0,len(topology)-1):
                assert pparams[i].W.shape == (topology[i],topology[i+1]), \
                    "W of layer %i wrong shape" % i
                assert pparams[i].b.shape == (topology[i+1],), \
                    "b of layer %i wrong shape" % i
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
                self.layers.append(lyr)
            
            # Initialise weights of each constructed layer
            init_lyr = Initialisation()
            for lyr in self.layers:
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
                self.layers.append(lyr)
                print(lyr, "built")
    
    
    
    def load_data(self, dataset):
        '''Load the dataset, which must be in pickled form. Will want to add a
        database retrieval function later
        
        :type dataset: string
        :param dataset: path to dataset directory
        '''
        
        print('Loading data')
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

####### NEED A SHARED DATASET METHOD!!!!!!!!!

    
    
    def pretrain(
        self, 
        method='AE',
        loss='SE',
        regulariser=('CAE'),
        optimiser='SDG',
        momentum='0.1',
        scheduler='ED'
        ):
        self.method = method
        self.loss = loss
        self.regulariser = regulariser
        self.optimiser = optimiser
        self.momentum = momentum
        self.scheduler = scheduler
        
        train = Train()
        
        for lyr in self.layers:
            train.train_layer(lyr)
        
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    