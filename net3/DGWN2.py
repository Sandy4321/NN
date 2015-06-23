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
        #self.num_c = args['num_components']
        
        self.b = [] # Neuron biases
        self.M = [] # Connection weight means
        self.R = [] # Connection weight variances (S = log(1+exp(R)))
        self.spl = [] # Sample shape
        self._params = []
        for i in numpy.arange(self.num_layers):
            # Biases initialized from zero
            b_value = 0.1*numpy.ones((self.ls[i+1],))[:,numpy.newaxis]
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            # Connection weight means initialized from zero
            #pre_coeff = numpy.sqrt(4./(self.ls[i+1] + self.ls[i]))
            pre_coeff = 0
            M_value = pre_coeff*numpy.random.randn(self.ls[i+1],self.ls[i])
            M_value = numpy.asarray(M_value, dtype=Tconf.floatX)
            Mname = 'M' + str(i)
            self.M.append(TsharedX(M_value, Mname, borrow=True))
            # Xavier initialization
            pre_coeff = 0.04/(self.ls[i+1] + self.ls[i])
            coeff = numpy.log(numpy.exp(numpy.sqrt(pre_coeff))-1.)
            R_value = coeff*numpy.ones((self.ls[i+1],self.ls[i]))
            R_value = numpy.asarray(R_value, dtype=Tconf.floatX)
            Rname = 'R' + str(i)
            self.R.append(TsharedX(R_value, Rname, borrow=True))
            # The mixing component mask
            
        for M, R, b in zip(self.M, self.R, self.b):
            self._params.append(M)
            self._params.append(R)
            self._params.append(b)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        '''
        M = T.dot(self.M[layer],X) 
        S = T.sqrt(T.dot(T.log(1 + T.exp(self.R[layer]))**2,X**2))
        E = self.gaussian_sampler(S.shape)
        H = M + S*E + self.b[layer]
        '''
        M = self.M[layer]
        S = T.log(1 + T.exp(self.R[layer]))
        b = self.b[layer]
        # Broadcasting rules
        M = M.dimshuffle(0,'x',1)
        S = S.dimshuffle(0,'x',1)
        X = X.dimshuffle('x',1,0)
        # Sample weights
        H = []
        for i in numpy.arange(10):
            E = self.gaussian_sampler((S.shape[0],10,S.shape[1]))
            W = M + (S*E)
            # Forwardprop
            H.append(T.sum(W*X[:,i*10:(i+1)*10,:], axis=2) + b)
        H = T.concatenate(H, axis=1)
        # Nonlinearity
        if nonlinearity == 'ReLU':
            f = lambda x : (x > 0) * x
        elif nonlinearity == 'SoftMax':
            f = Tnet.softmax
        elif nonlinearity == 'linear':
            f = lambda x : x
        else:
            print('Invalid nonlinearity')
            sys.exit(1)
        return f(H)
    
    def regularisation(self):
        '''Compute the regularisation'''
        reg = 0.
        for layer in numpy.arange(len(self.M)):
            S2 = T.log(1. + T.exp(self.R[layer]))
            P = T.sqrt(self.prior_variance)
            M = self.M[layer]
            reg += T.sum(T.log(S2/P) - 0.5 + 0.5*(((P**2) + (M**2))/(S2**2)))
        return reg
    
    def predict(self, X, args):
        '''Full MLP'''
        self.dropout_dict = args['dropout_dict']
        if 'num_samples' in args:
            if args['num_samples'] > 0:
                X = self.extra_samples(X,args)
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i, args)
        if args['mode'] == 'training':
            X = (X, self.regularisation())
        elif args['mode'] == 'validation':
            X = (X,)
        return X
        
    def gaussian_sampler(self, size):
        '''Return a standard gaussian vector'''
        smrg = MRG_RandomStreams(seed=235)
        rng = smrg.normal(size=size)
        return rng
    
    def integer_sampler(self, layer, size, num_c):
        '''Return a mask of ints to sample the mixing components'''
        smrg = MRG_RandomStreams(seed=234)
        rng = smrg.uniform(size=size)
        rng = T.floor(rng * num_c)
        return rng
    
    def extra_samples(self, X, args):
        '''Make parallel copies of the data'''
        mode = args['mode']
        n = args['num_samples']
        Y = T.concatenate([X,]*args['num_samples'], axis=1)
        print('Mode: %s, Number of samples: %i' % (mode, n))
        return Y

            
        
'''
TODO:
- EXPLORE INITIALISATION
- COMBINE WITH DROPOUT
- DEEP MoEs
- SPARSE CONNECTIONS
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    