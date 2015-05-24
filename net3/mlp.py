'''Multilayer perceptron model'''


__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import sys, time

import cPickle
import gzip
import numpy
import theano.tensor as T
import theano.tensor.nnet as Tnet

from draw_beta import Draw_beta
from matplotlib import pylab
from matplotlib import pyplot as plt
from theano import config as Tconf
from theano import function as Tfunction
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams


class Mlp():
    def __init__(self, args):
        '''Construct the MLP expression graph'''
        self.ls = args['layer_sizes']
        self.num_layers = len(self.ls) - 1
        self.dropout_dict = args['dropout_dict']
        
        self.W = [] # Connection weights
        self.b = [] # Biases
        self.q = [] # Dropout rates/prior
        self.G = [] # Dropout masks
        self.S = [] # Sparsity masks
        self._params = []
        for i in numpy.arange(self.num_layers):
            if 'connectivity' in args:
                beta = args['connectivity'][i]
            #Connection weights
            coeff = numpy.sqrt(6/(beta*(self.ls[i] + (self.ls[i+1]))))
            W_value = 2*coeff*(numpy.random.uniform(size=(self.ls[i+1],
                                                          self.ls[i]))-0.5)
            W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
            Wname = 'W' + str(i)
            self.W.append(TsharedX(W_value, Wname, borrow=True))
            # Biases
            b_value = 0.5*numpy.ones((self.ls[i+1],))[:,numpy.newaxis]
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            # Dropout/connect
            name = 'layer' + str(i)
            vname = 'drop' + str(i)
            if self.dropout_dict != None:
                if name in self.dropout_dict:
                    sub_dict = self.dropout_dict[name]
                    # Initialise q to some values
                    q_value = TsharedX(sub_dict['values'], vname,
                                       broadcastable=(False,True))
                    self.q.append(q_value)
            
            # Sparsity
            if (args['sparsity'] != None) and (i < self.num_layers - 1):
                sp = args['sparsity']
                sname = 'sparse' + str(i)
                sparse_mask = numpy.random.rand(self.ls[i+1],self.ls[i])<(1-sp)
                sparse_mask = TsharedX(sparse_mask.astype(Tconf.floatX),
                                      sname, borrow=True)
                self.S.append(sparse_mask)
            
        for W, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        name = 'layer' + str(layer)
        # Sparsity
        if (args['sparsity'] != None) and (layer < self.num_layers - 1):
            W = self.W[layer]*self.S[layer]
        else:
            W = self.W[layer]
            
        # Dropout
        if self.dropout_dict == None:
            pre_act = T.dot(W, X) + self.b[layer]
        elif name in self.dropout_dict:
            G = self.dropout(layer, X.shape)
            self.G.append(G > 0)                    # To access mask values
            pre_act = T.dot(W, X*G) + self.b[layer]
        else:
            pre_act = T.dot(W, X) + self.b[layer]
        
        # Nonlinearity
        if nonlinearity == 'ReLU':
            s = lambda x : (x > 0) * x
        elif nonlinearity == 'SoftMax':
            s = Tnet.softmax
        elif nonlinearity == 'wrapped_ReLU':
            period = args['period']
            b = args['deadband']
            pre_act = pre_act % period
            s = lambda x : T.minimum(((x - b) > 0) * (x - b), ((period - x - b) > 0) * (period - x - b))
        else:
            print('Invalid nonlinearity')
            sys.exit(1)
            
        return s(pre_act) 
    
    def predict(self, X, args):
        '''Full MLP'''
        self.dropout_dict = args['dropout_dict']
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i, args)
        return X
    
    def dropout(self, layer, size):
        '''Return a random dropout vector'''
        name = 'layer' + str(layer)
        if name in self.dropout_dict:
            sub_dict = self.dropout_dict[name]
            cseed = sub_dict['seed']
            # Construct RNG
            smrg = MRG_RandomStreams(seed=cseed)
            rng = smrg.uniform(size=size)
            # Evaluate RNG
            dropmult = (rng < self.q[layer]) / self.q[layer]
        return dropmult
    
def layer_from_sparsity(N, Y, T, a, b, c):
    '''Compute the hidden layer sizes'''
    # N input, Y output, T total weights (roughly)
    # a input connectivity, b core connectivity, c number core layers
    P = a*N + Y
    H = (numpy.sqrt((P**2) + 4*b*T) - P)/(2*b*c)
    H = numpy.floor(H)
    return int(H)

def total_weights(neurons):
    '''Compute the total number of weights in the network'''
    T = 0
    for i in numpy.arange(len(neurons)-1):
        T += neurons[i]*neurons[i+1]
    return int(T)

def write_neurons(N, H, Y, c):
    '''Write a neuron list'''
    L = []
    L.append(int(N))
    for i in numpy.arange(c+1):
        L.append(int(H))
    L.append(int(Y))
    return L
        
'''
TODO:
- LAYERWISE DROPOUT
- GPU BETA DISTRIBUTION SAMPLING
- VARIATIONAL BETA SCHEME
- LOCAL EXPECTATION GRADIENTS
- GAUSSIAN DROPOUT
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    