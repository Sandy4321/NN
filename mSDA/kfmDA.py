'''
kernel fmDA
'''

import numpy as np
import numpy.random as rp
import time
import cPickle
import gzip
import utils
from PIL import Image
import sys


class kfmDA:
    def __init__(self):
        pass
    
    def load(self, address):
        f = gzip.open(address, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return (train_set, valid_set, test_set)
    
    
    
    def SDA(self, machine, train_data, H):
        '''
        fmSDA builds a stack of mDAs
        
        :type machine:      string
        :param machine:     the machine type
        
        :type train_data:   numpy array
        :param train_data:  data stored columnwise
        
        :type k:            int
        :param k:           number of layers
        
        :type H:            int
        :param H:           number of units per layer
        
        We really need to decide whether to backpropagate the targets
        through a linear or inverted nonlinear decoder.
        '''
        weights     = []
        enc_bias    = []
        dec_bias    = []
        d           = train_data.shape[0]
        hidden_inp  = self.forward(np.eye(d), np.zeros(d), train_data, 'tanh')
        hidden_tgt  = train_data
        start = time.time()
       
        assert len(H) > 0
        for h in H:
            assert h > 0
            assert h % 1 == 0
        k = len(H)
        
        for i in xrange(k-1):
            print('Building layer %i' % i)
            if machine == 'underAE':
                B, bE, bD = self.underAE(hidden_inp, hidden_tgt, H[i])
                weights.append(B)
                enc_bias.append(bE)
                dec_bias.append(bD)
                hidden_inp = self.forward(weights[i], enc_bias[i], hidden_inp, 'tanh')
                hidden_tgt = self.backward(weights[i],dec_bias[i], hidden_tgt, 'linear')
                print('Elapsed time: %04f' % (time.time()-start,))
            else:
                print('Invalid machine')
                sys.exit(1)
            
            
        print('Building layer %i' % (k-1,))
        if machine == 'underAE':
                B, bE, bD = self.underAE(hidden_inp, hidden_tgt, H[k-1])
                weights.append(B)
                enc_bias.append(bE)
                dec_bias.append(bD)
        else:
            print('Invalid machine')
            sys.exit(1)
        # No need to put through nonlinearity again
        print('Elapsed time: %04f' % (time.time()-start,))
        
        return (weights, enc_bias, dec_bias)
    
    
    
    def forward(self, weights, bias, data, nonlinearity):
        pre_act = np.dot(weights, data) + bias[:,np.newaxis]
        if nonlinearity == 'tanh':
            h = np.tanh(pre_act)
        elif nonlinearity == 'linear':
            h = pre_act
        else:
            print('Invalid machine')
            sys.exit(1)
        return h
    
    
    
    def backward(self, weights, bias, data, nonlinearity):
        ##########CHECK THIS##############
        W = weights.T
        d = data - bias[:,np.newaxis]
        print W.shape, d.shape
        pre_act = np.linalg.lstsq(W, d)[0]
        if nonlinearity == 'linear':
            h = pre_act
        elif nonlinearity == 'arctanh':
            h = np.arctanh(pre_act)
        else:
            print('Invalid machine')
            sys.exit(1)
        print h.shape
        return h
    


    
    def map(self, test_data, params):
        '''
        map passes data through the DA
        
        :type test_data:    numpy array
        :param test_data:   the test data stored column-wise
        
        :type params:       list of numpy arrays
        :param params:      the parameters of the network
        '''
        # Build network
        weights, enc_bias, dec_bias = params
        start = time.time()
        hidden_inp  = test_data
        n           = test_data.shape[1]
        k           = len(weights)
        d           = test_data.shape[0]
        hidden_inp  = self.forward(np.eye(d), np.zeros(d), test_data, 'tanh')
        # Encoder
        for i in xrange(k-1):
            print('Propagating through layer %i' % i)
            hidden_inp = self.forward(weights[i], enc_bias[i], hidden_inp, 'tanh')
        # Middle layer is always linear
        print('Propagating through layer %i' % (k-1,))
        hidden_inp = self.forward(weights[k-1], enc_bias[k-1], hidden_inp, 'linear')
        # Decoder
        for i in xrange(k-1):
            print('Propagating through layer %i' % (i+k,))
            hidden_inp = self.forward(weights[k-i-1].T, dec_bias[k-i-1], hidden_inp, 'linear')
        print('Propagating through layer %i' % (2*k-1,))
        output = self.forward(weights[0].T, dec_bias[0], hidden_inp, 'linear')
        
        print('Elapsed time: %04f' % (time.time()-start,))
        return output    
    
    
    
    def underAE(self, X, Y, H):
        '''
        An undercomplete autoencoder with H hidden variables. We
        are going to have to use unbiased corruption for what follows.
        
        :type X:    numpy array
        :param X:   hidden layer input
        
        :type Y:    numpy array
        :param Y:   overall target
        
        :type H:    int
        :param H:   number of hidden units
        
        RETURNS
        W: weight matrix with biases in rightmost column
        b: biases of decoder
        '''
        assert H != None
        reg     = 1e-3
        mean    = 0.4
        variance= 0.04
        nu      = (mean*(1-mean)/variance)-1
        alpha   = mean*nu
        beta    = (1-mean)*nu
        while (alpha - 1.)**2 < 0.001:
            alpha += 0.01
        print('alpha: %0.3g, beta: %0.3g' % (alpha, beta))
        
        # Zero mean everything
        Y       = X
        
        meanX   = np.mean(X,axis=1)
        meanY   = np.mean(Y,axis=1)
        X0      = X - meanX[:,np.newaxis]
        Y0      = Y - meanY[:,np.newaxis]
        b       = Y0.mean(axis=1)
        d, n    = X0.shape
        
        # Natural statistics of the corruption
        D0      = X0
        Dbar    = np.dot(Y0,D0.T) + np.dot(Y0,D0.T).T
        
        # Spectral decompositon of the natural statistics
        Vclean  = np.dot(X0,X0.T)
        priorf  = beta*(alpha+beta-1)/((alpha+beta)*(alpha-1))
        V       = Vclean + priorf*np.diag(np.diag(Vclean))
        Vbar    = V + V.T
        U, L    = np.linalg.svd(Vbar, full_matrices=True)[0:2]
        UH      = U[:,:H]
        LH      = L[:H]
        LHsh    = LH.shape[0]
        
        # Forward weight matrix
        MH      = 1./( LH[:,np.newaxis] + LH[np.newaxis,:] + reg*np.ones((LHsh,LHsh)))
        Dproj   = 2*np.dot(UH.T,np.dot(Dbar,UH))
        Wbar    = MH*Dproj
        B       = np.dot(np.linalg.cholesky(Wbar + reg*np.eye(LHsh)),UH.T)
        bE      = np.linalg.lstsq(B.T,b)[0]
        bD      = b - np.dot(B.T,bE)
        
        return (B,bE,bD)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   