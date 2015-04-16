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
    
    def reconstruct(self, X, B, bE, bD):
        '''Reconstruct a single layer'''
        H = self.encode(X, B, bE)
        return self.decode(H, B, bD)
    
    def encode(self, X, B, bE):
        '''Encode a single layer'''
        return np.dot(B,X) + bE[:,np.newaxis]
        
    def decode(self, H, B, bD):
        '''Decoder a single layer'''
        return np.dot(B.T,H) + bD[:,np.newaxis]
    
    def underfmDA(self, X, Y, H):
        '''An undercomplete/columnar fmDA layer'''
        # Recall data is stored column-wise
        nvis = X.shape[0]
        nhid = H
        assert nvis >= nhid
        
        #print('Zeroing')
        # Zero mean everything - we'll push these onto the biases later
        meanX = np.mean(X,axis=1)
        meanY = np.mean(Y,axis=1)
        X0 = X - meanX[:,np.newaxis]
        Y0 = Y - meanY[:,np.newaxis]
        
        #print('Computing natural statistics')
        # Natural statistics of the corruption
        D = X0
        temp = np.dot(Y0,D.T)
        D_ =  temp + temp.T
        
        #print('Spectral decomposition')
        # Spectral decompositon of the natural statistics
        alpha = 2 # need a better way of implementing
        beta = 2
        coeff = beta*(alpha+beta-1)/((alpha+beta)*(alpha-1))
        
        V = np.dot(X0,X0.T)
        V += coeff*np.diag(np.diag(V))
        V_ = V + V.T
        U, L = np.linalg.svd(V_, full_matrices=True)[0:2]
        UH = U[:,:H]    # Eigenvectors
        LH = L[:H]      # Eigenvalues
        
        #print('Computing weights')
        # Forward weight matrix
        MH = 1./( LH[:,np.newaxis] + LH[np.newaxis,:] + 1e-3*np.ones((H,H)))
        Dproj = 2*np.dot(UH.T,np.dot(D_,UH))
        Wbar = MH*Dproj
        # Encoder matrix
        B = np.dot(np.linalg.cholesky(Wbar + 1e-3*np.eye(H)),UH.T)
        
        #print('Computing biases')
        # Biases
        c = np.dot(B.T,np.dot(B,meanX)) + meanY
        # Encoder bias
        bE = np.dot(B,c)
        # Decoder bias
        bD = c - np.dot(B.T,bE)
        
        return (B,bE,bD)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   