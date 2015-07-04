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
from theano import sparse as Tsp
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams


class Dgwn():
    def __init__(self, args):
        '''Construct the MLP expression graph'''
        self.ls = args['layer_sizes']
        self.num_layers = len(self.ls) - 1
        self.dropout_dict = args['dropout_dict']
        self.prior_variance = args['prior_variance']
        prior = args['prior']
        #self.num_c = args['num_components']
        
        self.M = [] # Connection weight means
        self.R = [] # Connection weight variances (S = log(1+exp(R)))
        self._params = []
        for i in numpy.arange(self.num_layers):
            # Connection weight means initialized from zero
            #pre_coeff = numpy.sqrt(4./(self.ls[i+1] + self.ls[i]))
            pre_coeff = 0.
            M_value = pre_coeff*numpy.random.randn(self.ls[i+1],self.ls[i]+1)
            M_value = numpy.asarray(M_value, dtype=Tconf.floatX)
            Mname = 'M' + str(i)
            self.M.append(TsharedX(M_value, Mname, borrow=True))
            # Xavier initialization
            if prior in ('Gaussian', 'Uniform'):
                pre_coeff = 2./(self.ls[i+1] + self.ls[i])
                coeff = numpy.log(numpy.exp(numpy.sqrt(pre_coeff))-1.)
            elif prior in ('DropConnect'):
                coeff = 0.
            # For tests only
            coeff = numpy.log(numpy.exp(numpy.sqrt(args['prior_variance']))-1.)
            R_value = coeff*numpy.ones((self.ls[i+1],self.ls[i]+1))
            R_value = numpy.asarray(R_value, dtype=Tconf.floatX)
            Rname = 'R' + str(i)
            self.R.append(TsharedX(R_value, Rname, borrow=True))
            # The mixing component mask
            
        for M, R in zip(self.M, self.R):
            self._params.append(M)
            self._params.append(R)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        prior = args['prior']
        b = T.ones_like(X[0,:]).dimshuffle('x',0)
        X = T.concatenate([X,b],axis=0)
        if hasattr(self, 'pruned'):
            if self.pruned:
                M = Tsp.basic.dot(self.M[layer],X)
                s = Tsp.sqr(Tsp.structured_log(Tsp.structured_add(Tsp.structured_exp(self.R[layer]),1.)))
                S = T.sqrt(Tsp.basic.dot(s,X**2))
        else:
            if prior in ('Gaussian','Uniform'):
                M = T.dot(self.M[layer],X) 
                S = T.sqrt(T.dot(T.log(1 + T.exp(self.R[layer]))**2,X**2))
            elif prior in ('DropConnect',):
                m = self.M[layer]
                p = 1./(1. + T.exp(self.R[layer]))
                M = T.dot(p*m,X)
                S = T.sqrt(T.dot(p*(1.-p)*(m**2),X**2))
        E = self.gaussian_sampler(layer, S.shape)
        H = M + S*E 
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
    
    def regularisation(self, args):
        '''Compute the regularisation'''
        reg = 0.
        prior = args['prior']
        for layer in numpy.arange(len(self.M)):
            S2 = T.log(1. + T.exp(self.R[layer]))
            P = T.sqrt(self.prior_variance)
            M = self.M[layer]
            if prior == 'Gaussian':
                reg += T.sum(T.log(S2/P) - 0.5 + 0.5*(((P**2) + (M**2))/(S2**2)))
            elif prior == 'Uniform':
                reg += -0.5*T.sum(T.log(S2))
            elif prior == 'DropConnect':
                reg += 0.*T.sum(T.log(S2))
            else:
                print('Invalid prior')
                sys.exit(1)
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
            X = (X, self.regularisation(args))
        elif args['mode'] == 'validation':
            X = (X,)
        return X
        
    def gaussian_sampler(self, layer, size):
        '''Return a standard gaussian vector'''
        smrg = MRG_RandomStreams()
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
    
    def load_params(self, params, args):
        '''Load the pickled network'''
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
            bname = 'b' + str(i)
            j = [j for j, param in enumerate(params) if bname == param.name][0]
            b_value = params[j].get_value()
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            # Connection weight means initialized from zero
            Mname = 'M' + str(i)
            j = [j for j, param in enumerate(params) if Mname == param.name][0]
            M_value = params[j].get_value()
            M_value = numpy.asarray(M_value, dtype=Tconf.floatX)
            self.M.append(TsharedX(M_value, Mname, borrow=True))
            # Connection weight root variances initialized from prior
            Rname = 'R' + str(i)
            j = [j for j, param in enumerate(params) if Rname == param.name][0]
            R_value = params[j].get_value()
            R_value = numpy.asarray(R_value, dtype=Tconf.floatX)
            self.R.append(TsharedX(R_value, Rname, borrow=True))
            
        for M, R, b in zip(self.M, self.R, self.b):
            self._params.append(M)
            self._params.append(R)
            self._params.append(b)
    
    def prune(self, proportion, scheme):
        '''Prune the weights according to the prefered scheme'''
        SNR = []
        # Cycle through layers
        for layer in numpy.arange(self.num_layers):
            Mname = 'M' + str(layer)
            j = [j for j, param in enumerate(self._params) if Mname == param.name][0]
            M_value = self._params[j].get_value()
            Rname = 'R' + str(layer)
            j = [j for j, param in enumerate(self._params) if Rname == param.name][0]
            R_value = self._params[j].get_value()
            S_value = numpy.log(1. + numpy.exp(R_value))
            if scheme == 'SNR':
                snr = numpy.log(1e-6 + numpy.abs(M_value)/S_value)
            elif scheme == 'KL':
                snr = (M_value/S_value)**2 + numpy.log(S_value)
                snr_min = numpy.amin(snr)
                snr = numpy.log(snr - snr_min + 1e-6)
            SNR.append(snr)
        hist, bin_edges = self.cumhist(SNR, 1000)
        # Find cutoff value
        idx = (hist > proportion)
        bin_edges = bin_edges[1:]
        cutoff = numpy.compress(idx, bin_edges)
        cutoff = numpy.amin(cutoff)
        self.masks = []
        for snr in SNR:
            self.masks.append(snr > cutoff)
        self.pruned = True
        self.to_csc()

    def cumhist(self, SNR, nbins):
        '''Return normalised cumulative histogram of SNR'''
        SNR = numpy.hstack([snr.flatten() for snr in SNR])
        # Histogram of SNRs
        hist, bin_edges = numpy.histogram(SNR, bins=nbins)
        hist = numpy.cumsum(hist)
        hist = hist/(hist[-1]*1.)
        return (hist, bin_edges)
    
    def to_csc(self):
        '''Convert the parameters to sparse matrix form'''
        for layer in numpy.arange(len(self.masks)):
            M = self.M[layer]*self.masks[layer]
            S = T.log(1 + T.exp(self.R[layer]))*self.masks[layer]
            spM = Tsp.csc_from_dense(M)
            spS = Tsp.structured_add(Tsp.csc_from_dense(S),-1.)
            spR = Tsp.structured_log(Tsp.structured_exp(spS))
            self.M[layer] = spM
            self.R[layer] = spR
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
