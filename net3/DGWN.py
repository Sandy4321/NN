'''Deep Gaussian Weight Network'''


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

from matplotlib import pylab
from matplotlib import pyplot as plt
from theano import config as Tconf
from theano import function as Tfunction
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams


class Dgwn():
    def __init__(self, args):
        '''Construct the MLP expression graph'''
        self.ls = args['layer_sizes']
        self.num_layers = len(self.ls) - 1
        self.dropout_dict = args['dropout_dict']
        
        self.b = [] # Neuron biases
        self.M = [] # Connection weight means
        self.R = [] # Connection weight variances (S = log(1+exp(R)))
        self._params = []
        for i in numpy.arange(self.num_layers):
            # Biases
            b_value = 0.5*numpy.ones((self.ls[i+1],))[:,numpy.newaxis]
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            # Connection weight means
            coeff = numpy.sqrt(6/(self.ls[i] + self.ls[i+1]))
            M_value = 2*coeff*(numpy.random.uniform(size=(self.ls[i+1],
                                                          self.ls[i]))-0.5)
            M_value = numpy.asarray(M_value, dtype=Tconf.floatX)
            Mname = 'M' + str(i)
            self.M.append(TsharedX(M_value, Mname, borrow=True))
            # Connection weight variances
            coeff = numpy.sqrt(6/(self.ls[i] + self.ls[i+1]))
            R_value = 2*coeff*(numpy.random.uniform(size=(self.ls[i+1],
                                                          self.ls[i]))-0.5)
            R_value = numpy.asarray(M_value, dtype=Tconf.floatX)
            Rname = 'R' + str(i)
            self.R.append(TsharedX(R_value, Rname, borrow=True))
            
        for M, b in zip(self.W, self.b):
            self._params.append(M)
            self._params.append(R)
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
        else:
            print('Invalid nonlinearity')
            sys.exit(1)
            
        return s(pre_act) 
    
    def predict(self, X, args):
        '''Full MLP'''
        self.dropout_dict = args['dropout_dict']
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i, args)
            if args['cov'] == True:
                self.X[i] = X
                self.XXT[i] = T.dot(X,X.T)
        return X
    

        
'''
TODO:
- WRITE THE BASIC PROGRAMME
- EXPLORE INITIALISATION
- COMBINE WITH DROPOUT
- DEEP MoEs
- SPARSE CONNECTIONS
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    