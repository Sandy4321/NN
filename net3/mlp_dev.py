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
import theano.sparse as Tsp

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
        if hasattr(self, 'pruned'):
            if self.pruned:
                if self.dropout_dict == None:
                    pre_act = Tsp.basic.dot(self.W[layer],X) + self.b[layer]
                elif name in self.dropout_dict:
                    G = self.dropout(layer, X.shape)
                    self.G.append(G > 0)                    # To access mask values
                    pre_act = Tsp.basic.dot(self.W[layer],X*G) + self.b[layer]
                else:
                    pre_act = Tsp.basic.dot(self.W[layer],X) + self.b[layer]
        else:
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
        if 'num_samples' in args:
            if args['num_samples'] > 0:
                X = self.extra_samples(X,args)
        for i in numpy.arange(self.num_layers):
            if args['premean'] == True:
                self.X[i] = X
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
        self.X = []
        self.X.append([])
        self.X = self.X*self.num_layers
        
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
            
        for W, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
    
    def prune(self, proportion, scheme, reweight=None):
        '''Prune the weights according to the prefered scheme'''
        SNR = []
        # Cycle through layers
        for layer in numpy.arange(self.num_layers):
            Wname = 'W' + str(layer)
            j = [j for j, param in enumerate(self._params) if Wname == param.name][0]
            W_value = self._params[j].get_value()
            if reweight != None:
                W_value = reweight[layer]*W_value
            if scheme == 'KL':
                snr = numpy.log(W_value**2 + 1e-6)
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
        
    def activation_pruning(self, proportion, scheme, reweight=None):
        '''Prune the weights according to the KL scheme'''
        SNR = []
        W = []
        R = []
        # Cycle through layers
        for layer in numpy.arange(self.num_layers):
            Wname = 'W' + str(layer)
            j = [j for j, param in enumerate(self._params) if Wname == param.name][0]
            W.append(self._params[j].get_value())
            if reweight != None:
                W[layer] = reweight[layer]*W[layer]
            R.append(0*W[layer] + 1)
        # Get cost
        sh = 0
        for layer in numpy.arange(self.num_layers):
            sh += W[layer].shape[0]
        C = numpy.zeros((sh,))
        # Set already zeros
        for layer in numpy.arange(self.num_layers):
            R[layer] = (R[layer] - (W[layer] == 0)).astype(numpy.bool)
        self.iterprune(W,R,C)
    
    def act_costs(self, W, R, C):
        '''Calculate weight costs'''
        i = 0
        # Current costs
        for layer in numpy.arange(self.num_layers):
            w = W[layer]
            r = R[layer]
            for row in numpy.arange(w.shape[0]):
                C[i] = self.acost(w[row,:], r[row,:])
                i += 1
        i = 0
        # Future costs
        for layer in numpy.arange(self.num_layers):
            w = W[layer]
            r = R[layer]
            for row in numpy.arange(w.shape[0]):
                # Get smallest nonzero element
                rhat = numpy.copy(r[row,:])
                y = numpy.abs(w[row,:]*rhat)
                argmin = numpy.argmin(y + (1e6 * (y==0)))
                rhat[argmin] = 0
                C[i] = self.acost(w[row,:], rhat) - C[i]
                i += 1
        return C
    
    def acost(self, y, r):
        '''The cost function'''
        k = (numpy.sum(y)/numpy.dot(y,r))**2
        m = 2.*(numpy.dot(y,r-1)/numpy.dot(y,r))**2
        cost = m + k - 1. - numpy.log(k)
        if m < 0:
            print('mwoops')
        if (k - 1. - numpy.log(k)) < 0:
            print('vwoops')
        if cost < 0:
            print('cwoops')
        return cost
    
    def iterprune(self, W, R, C):
        '''Prune'''
        for j in numpy.arange(1000):
            C = self.act_costs(W, R, C)
            am = numpy.argmin(C)
            i = 0
            for layer in numpy.arange(self.num_layers):
                w = W[layer]
                for row in numpy.arange(w.shape[0]):
                    i += 1
                    if i == am:
                        y = numpy.abs(w[row,:]*R[layer][row,:])
                        argmin = numpy.argmin(y + (1e6 * (y==0)))
                        R[layer][row,:] = 0
            print j


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
            W = self.W[layer]*self.masks[layer]
            spW = Tsp.csc_from_dense(W)
            self.W[layer] = spW
    
    def extra_samples(self, X, args):
        '''Make parallel copies of the data'''
        mode = args['mode']
        n = args['num_samples']
        Y = T.concatenate([X,]*args['num_samples'], axis=1)
        print('Mode: %s, Number of samples: %i' % (mode, n))
        return Y
    
    

        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    