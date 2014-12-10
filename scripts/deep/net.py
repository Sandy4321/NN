'''
The net class defines the overall topology of a neural networks, be it
directed or undirected. This is a very flexible setup which should give
the user a high degree of manipulative ability over the various aspects
of neural net training.
'''

import numpy as np

import theano
import theano.tensor as T

import layer

class Net(object):
    ''' Net class
    
    We can define a general network topology


