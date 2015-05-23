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
        self._params = []
        for i in numpy.arange(self.num_layers):
            #Connection weights
            coeff = numpy.sqrt(6/(self.ls[i] + (self.ls[i+1])))
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
            drop_type = args['drop_type']
            if self.dropout_dict != None:
                if name in self.dropout_dict:
                    sub_dict = self.dropout_dict[name]
                    # Initialise q to some values
                    if drop_type == 'dropout':
                        q_value = TsharedX(sub_dict['values'], vname,
                                           broadcastable=(False,True))
                    elif drop_type == 'dropconnect':
                        q_value = TsharedX(sub_dict['values'], vname)
                    self.q.append(q_value)
            
        for W, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        drop_type = args['drop_type']
        name = 'layer' + str(layer)
        if self.dropout_dict == None:
            W = self.W[layer]
            Xdrop = X
        elif name in self.dropout_dict:
            if drop_type == 'dropout':
                size = X.shape
                G = self.dropout(layer, size)
                self.G.append(G > 0)        # To access mask values
                W = self.W[layer]
                Xdrop = X*G
            elif drop_type == 'dropconnect':
                size = self.W[layer].shape
                G = self.dropconnect(layer, size)
                self.G.append(G > 0)        # To access mask values
                W = self.W[layer]*G
                Xdrop = X
                print('Dropconnect %i' % (layer,))
        else:
            W = self.W[layer]
            Xdrop = X
        pre_act = T.dot(W, Xdrop) + self.b[layer]
        
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
    
    def dropconnect(self, layer, size):
        '''Return a random dropout matrix'''
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
    
        
'''
TODO:
- GPU BETA DISTRIBUTION SAMPLING
- VARIATIONAL BETA SCHEME
- LOCAL EXPECTATION GRADIENTS
- GAUSSIAN DROPOUT
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    