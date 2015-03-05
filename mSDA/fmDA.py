'''
Script to implement mSDAs
'''
import numpy as np
import numpy.random as rp
import time
import cPickle
import gzip
import utils
from PIL import Image
import sys


nonlinearity = lambda x : np.tanh(x) # np.maximum(0,x)

class fmDA:
    def __init__(self):
        pass
    
    def load(self, address):
        f = gzip.open(address, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return (train_set, valid_set, test_set)
    
    
    
    def mDA(self, X, p):
        '''
        mDA builds a Minmin Chen style marginalised DAE
        
        :type X:    numpy array
        :param X:   data stored columnwise
        
        :type p:    float in [0,1]
        :param p:   probability of dropping
        '''
        # Valid corruption only
        assert p >= 0.
        assert p <= 1.
        # Add a bias
        X = np.vstack((X,np.ones((1,X.shape[1]))))
        # Dimension of data
        d = X.shape[0]
        # Corruption multiplier
        q = np.vstack((np.ones((d-1,1))*(1-p),1))
        # Scatter matrix
        S = np.dot(X,X.T)
        # Least squares solution
        Q = S*np.dot(q,q.T)
        np.fill_diagonal(Q,q.T*np.diag(S))
        P = S*q.T*np.ones((d,d))
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P[:-1,:].T).T
        return W
    
    
    
    def fmDA(self, X):
        '''
        fmDA builds a fully uniformly marginalised DAE
        
        :type X:    numpy array
        :param X:   data stored columnwise
        '''
        # Add a bias
        X = np.vstack((X,np.ones((1,X.shape[1]))))
        # Dimension of data + bias 1
        d = X.shape[0]
        # Scatter matrix
        S = np.dot(X,X.T)
        # Adaptive regulariser
        D = np.diag(np.sum(X**2,1))
        # Least squares solution
        P = 0.5*S
        Q = (S/3) + (D/6)
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P[:-1,:].T).T
        return W
    
    
    
    def rfmDA(self, X, Y):
        '''
        fmDA builds a fully uniformly marginalised DAE
        
        :type X:    numpy array
        :param X:   data stored columnwise
                
        :type Y:    numpy array
        :param Y:   target data stored columnwise
        '''
        # Add a bias
        X = np.vstack((X,np.ones((1,X.shape[1]))))
        Y = np.vstack((Y,np.ones((1,Y.shape[1]))))
        # Dimension of data + bias 1
        d = X.shape[0]
        # Scatter matrix
        S = np.dot(X,X.T)
        # Adaptive regulariser
        D = np.diag(np.sum(X**2,1))
        # Least squares solution
        P = 0.5*np.dot(Y,X.T)
        Q = (S/3) + (D/6)
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P[:-1,:].T).T
        return W
    
    
    
    def krfmDA(self, X, Y, kappa):
        '''
        fmDA builds a fully kappa marginalised DAE
        
        :type X:    numpy array
        :param X:   data stored columnwise
        
        :type Y:    numpy array
        :param Y:   target data stored columnwise
        '''
        # Add a bias
        X = np.vstack((X,np.ones((1,X.shape[1]))))
        Y = np.vstack((Y,np.ones((1,Y.shape[1]))))
        # Dimension of data + bias 1
        d = X.shape[0]
        # Scatter matrix
        S = np.dot(X,X.T)
        # Adaptive regulariser
        D = np.diag(np.sum(X**2,1))
        # Least squares solution
        P = np.dot(Y,X.T)
        Q = kappa*S + (1-kappa)*D
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P[:-1,:].T).T
        return W
    
    
    
    def SDA(self, machine, train_data, k, kappa=None):
        '''
        fmSDA builds a stack of mDAs
        
        :type machine:      string
        :param machine:     the machine type
        
        :type train_data:   numpy array
        :param train_data:  data stored columnwise
        
        :type k:            int
        :param k:           number of layers
        
        :type kappa:        float in [0,1]
        :param kappa:       kappa value for kappa marginalisation
        '''
        params      = []
        hidden_rep  = train_data
        n           = train_data.shape[1]
        start = time.time()
        if kappa is not None:
            assert kappa >= 0
            assert kappa <= 1
        
        for i in xrange(k-1):
            print('Building layer %i' % i)
            if machine == 'fmDA':
                params.append(self.fmDA(hidden_rep))
            elif machine == 'mDA':
                params.append(self.mDA(hidden_rep,0.5))
            elif machine == 'rfmDA':
                params.append(self.rfmDA(hidden_rep,train_data))
            elif machine == 'krfmDA':
                params.append(self.krfmDA(hidden_rep,train_data,kappa))
            else:
                print('Invalid machine')
                sys.exit(1)
            h_augmented = np.vstack((hidden_rep,np.ones((1,n))))
            hidden_rep  = nonlinearity(np.dot(params[i],h_augmented))
            print('Elapsed time: %04f' % (time.time()-start,))
            
        print('Building layer %i' % (k-1,))
        if machine == 'fmDA':
            params.append(self.fmDA(hidden_rep))
        elif machine == 'mDA':
            params.append(self.mDA(hidden_rep,0.5))
        elif machine == 'rfmDA':
            params.append(self.rfmDA(hidden_rep,train_data))
        elif machine == 'krfmDA':
                params.append(self.krfmDA(hidden_rep,train_data,kappa))
        else:
            print('Invalid machine')
            sys.exit(1)
        # No need to put through nonlinearity again
        print('Elapsed time: %04f' % (time.time()-start,))
        
        return params
    
    
    
    def SVDnet(self, train_data, k, w, kappa):
        '''
        SVDnet builds a krfmDA using the split architecture
        '''
        params      = []
        hidden_rep  = train_data
        n           = train_data.shape[1]
        start = time.time()
        if kappa is not None:
            assert kappa >= 0
            assert kappa <= 1
        
        assert len(w) is k
        
        for i in xrange(k-1):
            print('Building layer %i' % i)
            W       = self.krfmDA(hidden_rep, hidden_rep, kappa)
            A, B    = self.Wsplit(w[i], W)
            params.append((A,B))
            h_augmented = np.vstack((hidden_rep,np.ones((1,n))))
            hidden_rep  = nonlinearity(np.dot(B,h_augmented))
            print('Elapsed time: %04f' % (time.time()-start,))
            
        print('Building layer %i' % (k-1,))
        W       = self.krfmDA(hidden_rep, hidden_rep, kappa)
        A, B    = self.Wsplit(w[i], W)
        params.append((A,B))
        # No need to put through nonlinearity again
        print('Elapsed time: %04f' % (time.time()-start,))
        
        return params


    
    def map(self, test_data, params):
        '''
        map passes data through the DA
        
        :type test_data:    numpy array
        :param test_data:   the test data stored column-wise
        
        :type params:       list of numpy arrays
        :param params:      the parameters of the network
        '''
        # Build network
        hidden_rep  = test_data
        n           = test_data.shape[1]
        k           = len(params)
        start = time.time()
        for i in xrange(k-1):
            print('Propagating through layer %i' % i)
            h_augmented = np.vstack((hidden_rep,np.ones((1,n))))
            hidden_rep  = nonlinearity(np.dot(params[i],h_augmented))
        # Final layer has no nonlinearity
        h_augmented = np.vstack((hidden_rep,np.ones((1,n))))
        hidden_rep  = np.dot(params[-1],h_augmented)
        print('Elapsed time: %04f' % (time.time()-start,))
        return hidden_rep
        
        
        
    def test(self, test_data, params):
        '''
        test measures the error under the loss function
        
        :type test_data:    numpy array
        :param test_data:   the test data stored column-wise
        
        :type params:       list of numpy arrays
        :param params:      the parameters of the network
        '''
        # Build network - need to calculate across all noise levels
        hidden_rep  = self.map(test_data,params)
        n           = test_data.shape[1]
        loss        = 0.5*((test_data - hidden_rep)**2).sum()/n
        
        return loss
    
    
    
    def Wsplit(self, w, X):
        '''
        Wsplit computes the SVD of a matrix and returns two matrices corresponding
        to a decomposition using the w largest singular components
        
        N.B. will add incremental updates later
        
        :type w:    int
        :param w:   the number of singular values to keep
        
        :type X:    numpy array
        :param X:   the matrix to SVD
        '''
        u, s, v = np.linalg.svd(X)
        U       = u[:,:w]
        S       = np.diag(s[:w])
        V       = v[:w,:]
        
        A       = np.dot(U,S)
        B       = np.dot(S,V)
        
        return (A,B)
    
    

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   