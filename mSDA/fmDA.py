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
        fmDA builds a fully uniformly marginalised DAE with retargetting
        
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
        fmDA builds a fully kappa marginalised DAE with retargetting
        
        :type X:    numpy array
        :param X:   data stored columnwise
        
        :type Y:    numpy array
        :param Y:   target data stored columnwise
        '''
        assert kappa >= 0.
        assert kappa <= 1.
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
    
    
    
    def biasDA(self, X, Y, kappa):
        '''
        biasDA builds a fully kappa marginalised DAE with retargetting
        
        :type X:    numpy array
        :param X:   data stored columnwise
              
        :type Y:    numpy array
        :param Y:   target data stored columnwise
        
        :type kappa:    float in [0,1]
        :params kappa:  kappa regularisation constant
        '''
        assert kappa >= 0.
        assert kappa <= 1.
        # Add a bias
        X = np.vstack((X,np.ones((1,X.shape[1]))))
        # Dimension of data + bias 1
        d = X.shape[0]
        # Corruption multiplier
        Qn                  = np.ones((d,d))
        Qn[:(d-1),:(d-1)]   = Qn[:(d-1),:(d-1)]*kappa
        np.fill_diagonal(Qn,1.)
        # Scatter matrix
        S = np.dot(X,X.T)
        # Block form
        Q = S*Qn
        np.fill_diagonal(Q,np.diag(S))
        P = np.dot(Y,X.T)
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P.T).T
        return W
    
    
    
    def fmDAb(self, X, Y, c1, c2):
        '''
        biasDA builds a fully marginalised DAE with retargetting and
        correct biases
        
        :type X:    numpy array
        :param X:   data stored columnwise
              
        :type Y:    numpy array
        :param Y:   target data stored columnwise
        
        :type c1:   float in [0,1]
        :params c1: mean parameter
        
        :type c2:   float in [0,1]
        :param c2:  energy parameter
        '''
        kappa   = c2/c1
        assert kappa >= 0.
        assert kappa <= 1.
        # Add a bias
        Xs= X.shape
        Ys= Y.shape
        # Dimension of data + bias 1
        d = Xs[0]+1
        N = Xs[1]
        X = np.vstack((X,np.ones((1,N))))
        # Corruption multiplier
        Qn                  = c2*np.ones((d,d))
        np.fill_diagonal(Qn,c1)
        Qn[:,-1]            = c1
        Qn[-1,:]            = c1
        Qn[-1,-1]           = 1
        Pn                  = c1*np.ones((Ys[0],Xs[0]))
        Pn                  = np.hstack((Pn,np.ones((Ys[0],1))))
        # Scatter matrix
        S = np.dot(X,X.T)
        # Block form
        Q = S*Qn
        P = np.dot(Y,X.T)*Pn
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P.T).T
        return W
    
    
    def SDA(self, machine, train_data, k, kappa=None, c1=None, c2=None, p=None):
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
            assert kappa >= 0.
            assert kappa <= 1.
        
        if p is not None:
            assert p >= 0.
            assert p <= 1.
        
        for i in xrange(k-1):
            print('Building layer %i' % i)
            if machine == 'fmDA':
                params.append(self.fmDA(hidden_rep))
            elif machine == 'mDA':
                params.append(self.mDA(hidden_rep, p))
            elif machine == 'rfmDA':
                params.append(self.rfmDA(hidden_rep, train_data))
            elif machine == 'krfmDA':
                params.append(self.krfmDA(hidden_rep, train_data,kappa))
            elif machine == 'biasDA':
                params.append(self.biasDA(hidden_rep, train_data, kappa))
            elif machine == 'fmDAb':
                params.append(self.fmDAb(hidden_rep, train_data, c1, c2))
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
        elif machine == 'biasDA':
            params.append(self.biasDA(hidden_rep,train_data, kappa))
        elif machine == 'fmDAb':
                params.append(self.fmDAb(hidden_rep,train_data, c1, c2))
        else:
            print('Invalid machine')
            sys.exit(1)
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
    
    
    
    def biasDA(self, X, Y):
        '''
        mDA builds a Minmin Chen style marginalised DAE
        
        :type X:    numpy array
        :param X:   data stored columnwise
              
        :type Y:    numpy array
        :param Y:   target data stored columnwise
        '''
        c1 = 1./2
        c2 = 1./3

        # Add a bias
        X = np.vstack((X,np.ones((1,X.shape[1]))))
        Y = np.vstack((Y,np.ones((1,Y.shape[1]))))
        # Dimension of data + bias 1
        d = X.shape[0]
        

        # Corruption multiplier
        q1 = np.vstack((np.ones((d-1,1))*c1,1))
        Q2 = c2*np.ones((d-1,d-1))
        Q2 = np.hstack((Q2,c1*np.ones((d-1,1))))
        Q2 = np.vstack((Q2,q1.T))
        # Scatter matrix
        S = np.dot(X,X.T)
        # Least squares solution
        Q = S*Q2
        np.fill_diagonal(Q,q1.T*np.diag(S))
        P = np.dot(Y,X.T)*q1.T*np.ones((d,d))
        # Weights
        W = np.linalg.solve(Q.T+1e-5*np.eye(d),P[:-1,:].T).T
        return W
    
    
    
    def underAE(self, X, Y, H):
        '''
        An undercomplete autoencoder with H hidden variables. We
        are going to have to use unbiased corruption for what follows.
        
        Corruption is not used for now
        '''
        # Zero mean everything
        meanX   = np.mean(X,axis=1)
        meanY   = np.mean(Y,axis=1)
        X0      = X - meanX[:,np.newaxis]
        Y0      = Y - meanY[:,np.newaxis]
        
        D0      = X0
        Dbar    = np.dot(Y0,D0.T) + np.dot(Y0,D0.T).T
        
        V       = np.dot(X0,X0.T)
        

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   