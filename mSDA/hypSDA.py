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


logistic = lambda x : (1. / (1 + np.exp(-x)))

class fmDA:
    def __init__(self):
        pass
    
    def load(self, address):
        f = gzip.open(address, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return (train_set, valid_set, test_set)
    
    
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
    
    
    
    def fmSDA(self, machine, train_data, k):
        '''
        fmSDA builds a stack of mDAs
        
        :type machine:      string
        :param machine:     the machine type
        
        :type train_data:   numpy array
        :param train_data:  data stored columnwise
        
        :type k:            int
        :param k:           number of layers
        '''
        params      = []
        hidden_rep  = train_data
        n           = train_data.shape[1]
        start = time.time()
        for i in xrange(k-1):
            print('Building layer %i' % i)
            if machine == 'fmSDA':
                params.append(self.fmDA(hidden_rep))
            elif machine == 'mSDA':
                params.append(self.mDA(hidden_rep,0.5))
            else:
                print('Invalid machine')
                sys.exit(1)
            h_augmented = np.vstack((hidden_rep,np.ones((1,n))))
            hidden_rep  = logistic(np.dot(params[i],h_augmented))
            print('Elapsed time: %04f' % (time.time()-start,))
        print('Building layer %i' % (k-1,))
        params.append(self.fmDA(hidden_rep))
        print('Elapsed time: %04f' % (time.time()-start,))
        
        return params
    
    
    
    def test(self, test_data, params):
        '''
        test measures the error under the loss function
        
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
        for i in xrange(k):
            print('Propagating through layer %i' % i)
            h_augmented = np.vstack((hidden_rep,np.ones((1,n))))
            hidden_rep  = logistic(np.dot(params[i],h_augmented))
        print('Elapsed time: %04f' % (time.time()-start,))
        loss = 0.5*((test_data - hidden_rep)**2).sum()/n
        
        return loss
        
if __name__ == '__main__':
    fmda = fmDA()
    # Note the data is stored row-wise and the fmDA takes it column-wise
    print('Loading data')
    T, V, test = fmda.load('../net/data/mnist.pkl.gz')
    X       = np.vstack((T[0],V[0])).T
    Xtest   = test[0].T
    print('Computing layers')
    params_fmSDA= fmda.fmSDA('fmSDA',X,3)
    params_mSDA = fmda.fmSDA('mSDA',X,3)
    loss_fmSDA  = fmda.test(Xtest,params_fmSDA)
    loss_mSDA   = fmda.test(Xtest,params_mSDA)
    print loss_fmSDA, loss_mSDA
    
    
    
    '''
    num_imgs = 400
    index = rp.choice(W.shape[1], num_imgs, replace=False)
    img = W[:,index]
    image = Image.fromarray(utils.tile_raster_images(X=img.T,
             img_shape=(28,28), tile_shape=(20, 20),
             tile_spacing=(1, 1)))
    image.save('hypSDA.png')
    '''
   
   
   