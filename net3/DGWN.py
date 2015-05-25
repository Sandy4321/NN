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
        self.prior_variance = args['prior_variance']
        
        self.b = [] # Neuron biases
        self.M = [] # Connection weight means
        self.R = [] # Connection weight variances (S = log(1+exp(R)))
        self._params = []
        for i in numpy.arange(self.num_layers):
            # Biases
            b_value = 0.1*numpy.ones((self.ls[i+1],))[:,numpy.newaxis]
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            # Connection weight means
            coeff = self.prior_variance
            M_value = coeff*(numpy.random.rand(self.ls[i+1],self.ls[i]))
            M_value = numpy.asarray(M_value, dtype=Tconf.floatX)
            Mname = 'M' + str(i)
            self.M.append(TsharedX(M_value, Mname, borrow=True))
            # Connection weight root variances
            coeff = numpy.sqrt(2/(self.ls[i] + self.ls[i+1]))
            R_value = coeff*numpy.ones((self.ls[i+1],self.ls[i]))
            R_value = numpy.asarray(R_value, dtype=Tconf.floatX)
            Rname = 'R' + str(i)
            self.R.append(TsharedX(R_value, Rname, borrow=True))
            
        for M, R, b in zip(self.M, self.R, self.b):
            self._params.append(M)
            self._params.append(R)
            self._params.append(b)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        M = T.dot(self.M[layer],X)
        S = T.dot(T.log(1 + T.exp(self.R[layer])),X)
        E = self.gaussian_sampler(layer, S.shape)
        H = M + 0.01*S*E + self.b[layer]
        # Nonlinearity
        if nonlinearity == 'ReLU':
            f = lambda x : (x > 0) * x
        elif nonlinearity == 'SoftMax':
            f = Tnet.softmax
        else:
            print('Invalid nonlinearity')
            sys.exit(1)
        return f(H)
    
    def regularisation(self):
        '''Compute the regularisation'''
        KL = 0
        for M, R in zip(self.M, self.R):
            S2 = T.log(1 + T.exp(R))**2
            P2 = self.prior_variance
            KL += 0.5 * T.sum(1 + T.log(S2/P2) - ((M**2)/P2) - (S2/P2))
        return KL
    
    def predict(self, X, args):
        '''Full MLP'''
        self.dropout_dict = args['dropout_dict']
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i, args)
        if args['mode'] == 'training':
            X = (X, 0) # -self.regularisation()) 
        elif args['mode'] == 'validation':
            X = (X,)
        return X
        
    def gaussian_sampler(self, layer, size):
        '''Return a standard gaussian vector'''
        name = 'layer' + str(layer)
        smrg = MRG_RandomStreams(seed=234)
        rng = smrg.normal(size=size)
        return rng

        
'''
TODO:
- REGULARISATION
- NUMBER OF SAMPLES
- EXPLORE INITIALISATION
- COMBINE WITH DROPOUT
- DEEP MoEs
- SPARSE CONNECTIONS
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    