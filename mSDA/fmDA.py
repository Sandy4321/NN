"""Fully marginalised autoencoder fmda"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import gzip, sys, time

import numpy as np
import cPickle
import utils

class fmDA:
    def __init__(self):
        pass
    
    def load(self, address):
        f = gzip.open(address, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return (train_set, valid_set, test_set)
    
    def stack(self, machine, train_data, H):
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
        hidden_inp  = self.forward(np.eye(d), np.zeros(d), train_data, 'linear')
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
                hidden_inp = self.forward(weights[i], enc_bias[i], hidden_inp, 'linear')
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
    
    def underfmDA(self, X, Y, H):
        '''An undercomplete/columnar fmDA layer'''
        # Recall data is stored column-wise
        nvis = X.shape[0]
        nhid = H
        assert nvis >= nhid
        
        print('Zeroing')
        # Zero mean everything - we'll push these onto the biases later
        meanX = np.mean(X,axis=1)
        meanY = np.mean(Y,axis=1)
        X0 = X - meanX[:,np.newaxis]
        Y0 = Y - meanY[:,np.newaxis]
        
        print('Computing natural statistics')
        # Natural statistics of the corruption
        D = X0
        temp = np.dot(Y0,D.T)
        D_ =  temp + temp.T
        
        print('Spectral decomposition')
        # Spectral decompositon of the natural statistics
        alpha = 1.2 # need a better way of implementing
        beta = 0.8
        coeff = beta*(alpha+beta-1)/((alpha+beta)*(alpha-1))
        
        V = np.dot(X0,X0.T)
        V += coeff*np.diag(np.diag(V))
        V_ = V + V.T
        U, L = np.linalg.svd(V_, full_matrices=True)[0:2]
        UH = U[:,:H]    # Eigenvectors
        LH = L[:H]      # Eigenvalues
        
        print('Computing weights')
        # Forward weight matrix
        MH = 1./( LH[:,np.newaxis] + LH[np.newaxis,:] + 1e-3*np.ones((H,H)))
        Dproj = 2*np.dot(UH.T,np.dot(D_,UH))
        Wbar = MH*Dproj
        # Encoder matrix
        B = np.dot(np.linalg.cholesky(Wbar + 1e-3*np.eye(H)),UH.T)
        
        print('Computing biases')
        # Biases
        c = np.dot(B.T,np.dot(B,meanX)) + meanY
        # Encoder bias
        bE = np.dot(B,c)
        # Decoder bias
        bD = c - np.dot(B.T,bE)
        
        return (B,bE,bD)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   