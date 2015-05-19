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
        self.alpha = [] # Beta hyperparameter alpha
        self.beta = [] # Beta hyperparameter beta
        self.q = [] # Dropout rates
        self.G = {} # Dropout masks
        self._params = []
        self._hypparams = []
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
            # Dropout
            name = 'layer' + str(i)
            vname = 'dropout' + str(i)
            alphaname = 'alpha' + str(i)
            betaname = 'beta' + str(i)
            if self.dropout_dict != None:
                if name in self.dropout_dict:
                    # Dropout rates - this formulation means we must have
                    # dropout in every layer. Will need to change this later
                    sub_dict = self.dropout_dict[name]
                    ones = numpy.ones((sub_dict['values'].shape[0],1)).astype(Tconf.floatX)
                    a = sub_dict['prior_params'][0] * ones
                    b = sub_dict['prior_params'][1] * ones
                    alpha_value = TsharedX(a, alphaname)
                    beta_value = TsharedX(b, betaname)
                    # Initialise q to some values
                    q_value = TsharedX(sub_dict['values'], vname,
                                       broadcastable=(False,True))
                    self.q.append(q_value)
                    self.alpha.append(alpha_value)
                    self.beta.append(beta_value)
            
        for W, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
        for a, b in zip(self.alpha, self.beta):
            self._hypparams.append(a)
            self._hypparams.append(b)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        name = 'layer' + str(layer)
        if self.dropout_dict == None:
            Xdrop = X
        elif name in self.dropout_dict:
            size = X.shape
            # Sample q
            if args['variational_sample'] == True:
                a = self.alpha[layer]
                b = self.beta[layer]
                self.set_q(layer, a, b)
            G = self.dropout(layer, size)
            # Dropout masks need to be shared in order to be accessed
            Gname = 'mask' + str(layer)
            self.G[Gname] = G > 0   # The masks as binary!
            Xdrop = X*G
        else:
            Xdrop = X
        pre_act = T.dot(self.W[layer], Xdrop) + self.b[layer]
        
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
        vname = 'dropout' + str(layer)
        if name in self.dropout_dict:
            sub_dict = self.dropout_dict[name]
            cseed = sub_dict['seed']
            ctype = sub_dict['type']
            if ctype == 'unbiased':
                # Construct RNG
                smrg = MRG_RandomStreams(seed=cseed)
                rng = smrg.uniform(size=size)
                # Evaluate RNG
                dropmult = (rng < self.q[layer]) / self.q[layer]
        return dropmult
    
    def set_q(self, layer, a, b):
        '''Set the dropout probabilities in layer number layer to values'''
        # Construct RNG
        size = self.q[layer].shape
        cseed = 100
        smrg = MRG_RandomStreams(seed=cseed)
        u = smrg.uniform(size=size)
        q = Draw_beta()(a, b, u)[0]
        q = T.patternbroadcast(q, self.q[layer].broadcastable)
        self.q[layer] = q

        
'''
TODO:
- GPU BETA DISTRIBUTION SAMPLING
- VARIATIONAL BETA SCHEME
- LOCAL EXPECTATION GRADIENTS
- GAUSSIAN DROPOUT
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    