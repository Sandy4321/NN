'''Autoencoder model'''

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

class Autoencoder():
    def __init__(self, args):
        '''Construct the autoencoder expression graph'''

        self.ls = args['layer_sizes']
        self.num_layers = len(self.ls) - 1
        
        self.W = []
        self.b = []
        self.c = []
        self._params = []
        for i in numpy.arange(self.num_layers):
            coeff = numpy.sqrt(6/(self.ls[i] + (self.ls[i+1])))
            W_value = 2*coeff*(numpy.random.uniform(size=(self.ls[i+1],
                                                          self.ls[i]))-0.5)
            W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
            Wname = 'W' + str(i)
            self.W.append(TsharedX(W_value, Wname, borrow=True))
            
            b_value = 0.1*numpy.ones((self.ls[i+1],))[:,numpy.newaxis]
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            
            c_value = 0.1*numpy.ones((self.ls[i],))[:,numpy.newaxis]
            c_value = numpy.asarray(c_value, dtype=Tconf.floatX)
            cname = 'c' + str(i)
            self.c.append(TsharedX(c_value, cname, borrow=True,
                                   broadcastable=(False,True)))
        
        for W, b, c in zip(self.W, self.b, self.c):
            self._params.append(W)
            self._params.append(b)
            self._params.append(c)
        
        # Load the dropout variables
        self.dropout_dict = args['dropout_dict']
    
    def encode_layer(self, X, layer):
        '''Sigmoid encoder function for single layer'''          
        if self.dropout_dict == None:
            Xdrop = X
        else:
            size = X.shape
            Xdrop = X*self.dropout(layer, size)
        pre_act = T.dot(self.W[layer], Xdrop) + self.b[layer]  
        return (pre_act > 0) * pre_act 
    
    def decode_layer(self, h, layer):
        '''Linear decoder function for a single layer'''
        idx = self.num_layers - layer - 1
        lyr = self.num_layers + layer
        if self.dropout_dict == None:
            hdrop = h
        else:
            size = h.shape
            hdrop = h*self.dropout(lyr, size)
        pre_act = T.dot(self.W[idx].T, hdrop) + self.c[idx]
        return (pre_act > 0) * pre_act
    
    def encode(self, X):
        '''Full encoder'''
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i)
        return X
    
    def decode(self, h):
        '''Full decoder'''
        for i in numpy.arange(self.num_layers):
            h = self.decode_layer(h, i)
        return h

    def predict(self, X):
        '''Reconstruct input'''
        h = self.encode(X)
        return self.decode(h)
    
    def dropout(self, layer, size):
        '''Return a random dropout vector'''
        name = 'layer' + str(layer)
        vname = 'dropout' + str(layer)
        if name in self.dropout_dict:
            print name
            sub_dict = self.dropout_dict[name]
            cseed = sub_dict['seed']
            ctype = sub_dict['type']
            cvalues = TsharedX(sub_dict['values'], vname, broadcastable=(False, True))
            
            if ctype == 'unbiased':
                # Construct RNG
                smrg = MRG_RandomStreams(seed=cseed)
                rng = smrg.uniform(size=size)
                # Evaluate RNG
                dropmult = (rng < cvalues) / cvalues
        else:
            dropmult = 1.
        
        return dropmult
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    