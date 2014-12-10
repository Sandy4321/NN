'''
The net class defines the overall topology of a neural networks, be it
directed or undirected. This is a very flexible setup which should give
the user a high degree of manipulative ability over the various aspects
of neural net training.
'''

import numpy as np

import theano
import theano.tensor as T

from layer import Layer
from initialisation import Initialisation

class Net(object):
    ''' Net class
    
    We can define a general network topology. At the momemt masks are not
    supported and all hidden nodes MUST bind ONE input to ONE output
    '''
    
    def __init__(
        self,
        topology=(784, 500, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot',
        input_bias = False
        ):
        
        self.layers = []    # store layer objects
        
        assert len(nonlinearities) == len(topology) - 1
        self.nonlinearities = nonlinearities
        
        '''
        Now we simply loop through the elements of 'topology' and create a
        network layer with appropriate bound nonlinearities and initialised
        weights/biases.
        '''
        
        for i in np.arange(1,len(topology)):
            if (input_bias == True) and (i == 1):
                lyr = Layer(topology[i-1],topology[i], b_in = topology[i-1])
            
            lyr = Layer(topology[i-1],topology[i])
            self.layers.append(lyr)
        
        for lyr in self.layers:
            
        
        












if __name__ == '__main__':
    net = Net()