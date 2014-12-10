'''
Initialisation of all the network parameters is a tricky, heuristic
process which generally determines the overall performance of the system.
This class seeks to streamline some of the main initialisation procedures
as per Bengio, Glorot, Hinton and LeCun, so that initialisation can
be implemented via a clean and simple interface. Simply construct a
neural network and then call the initialisation class.
'''

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layer import Layer


class Initialise:
    ''' Initialise class
    
    Take in a layer object and an initialisation command and generate
    a set of suitable intial weights and biases
    '''
    
    def initialise(self, lyr, command, nonlinearity='sigmoid'):
        
        if command == 'None':
            pass
        elif command == 'Glorot':
            initial_W = np.zeros(shape=(n_out), dtype=theano.config.floatX)
            lyr.W.set_value(initial_W)
            