'''
Initialisation of all the network parameters is a tricky, heuristic
process which generally determines the overall performance of the system.
This class seeks to streamline some of the main initialisation procedures
as per Bengio, Glorot, Hinton and LeCun, so that initialisation can
be implemented via a clean and simple interface. Simply construct a
neural network and then call the initialisation class.
'''
import sys

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

from layer import Layer


class Initialisation:
    ''' Initialisation class
    
    Take in a layer object and an initialisation command and generate
    a set of suitable intial weights and biases
    '''
    
    def init_weights(self, lyr, command='Glorot', nonlinearity='sigmoid'):
        
        if command == 'None':
            pass
        elif command == 'Glorot':
            W_shape = lyr.W.get_value(borrow=True, return_internal_type=True).shape

            if nonlinearity == 'sigmoid':
                r = np.sqrt(6.0/(sum(W_shape)))
            elif nonlinearity == 'tanh':
                r = 4.0*np.sqrt(6.0/(sum(W_shape)))
            else:
                print 'Invalid nonlinearity'
                exit(1)
            
            np_rng = r*np.random.random_sample(size=W_shape)
            lyr.W.set_value(np_rng)
            