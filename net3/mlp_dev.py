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
        self.X = [] # Activity storage
        self.XXT = [] # Covariance storage
        self._params = []
        for i in numpy.arange(self.num_layers):
            if 'connectivity' in args:
                beta = args['connectivity'][i]
            else:
                beta = 1.
            # Covariance analysis
            if args['cov'] == True:
                self.X.append([])
                self.XXT.append([])
            # Connection weights
            coeff = numpy.sqrt(6/(beta*(self.ls[i] + self.ls[i+1])))
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
            
        for W, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        name = 'layer' + str(layer)
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
        return (X,T.zeros_like(X[0,0]))
    
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
    
    def load_params(self, params, args):
        '''Load the pickled network'''
        '''Construct the MLP expression graph'''
        self.ls = args['layer_sizes']
        self.num_layers = len(self.ls) - 1
        self.dropout_dict = args['dropout_dict']
        self.prior_variance = args['prior_variance']
        
        self.b = [] # Neuron biases
        self.W = [] # Connection weight means
        self._params = []
        for i in numpy.arange(self.num_layers):
            bname = 'b' + str(i)
            j = [j for j, param in enumerate(params) if bname == param.name][0]
            b_value = params[j].get_value()
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            self.b.append(TsharedX(b_value, bname, borrow=True,
                                   broadcastable=(False,True)))
            # Connection weight means initialized from zero
            Wname = 'W' + str(i)
            j = [j for j, param in enumerate(params) if Wname == param.name][0]
            W_value = params[j].get_value()
            W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
            self.W.append(TsharedX(W_value, Wname, borrow=True))
            
        for M, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
    
    def prune(self, proportion, scheme):
        '''Prune the weights according to the prefered scheme'''
        SNR = []
        # Cycle through layers
        for layer in numpy.arange(self.num_layers):
            Wname = 'W' + str(layer)
            j = [j for j, param in enumerate(self._params) if Wname == param.name][0]
            W_value = self._params[j].get_value()
            if scheme == 'KL':
                snr = numpy.log(W_value**2)
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
    
        

        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    