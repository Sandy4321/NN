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

def load(address):
    f = gzip.open(address, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)


def mDA(X):
    # Add a bias
    X = np.vstack((X,np.ones((1,X.shape[1]))))
    # Dimension of data + bias 1
    d = X.shape[0]
    # Corruption multiplier
    qP = np.vstack((np.ones((d-1,1))*0.5,1))
    qQ = np.ones((d-1,1))*(1./np.sqrt(3.))
    qQ = np.dot(qQ,qQ.T)
    qQ = np.vstack((qQ,qP[:-1].T))
    qQ = np.hstack((qQ,qP))
    print qQ
    # Scatter matrix
    S = np.dot(X,X.T)
    # Least squares solution
    Q = S*qQ
    np.fill_diagonal(Q,qP.T*np.diag(S))
    P = S*qP.T*np.ones((d,d))
    # Weights
    W = np.linalg.solve(Q.T+1e-5*np.eye(d),P[:-1,:].T).T
    return W
    
    
if __name__ == '__main__':
    # Note the data is stored row-wise and the mSDA takes it column-wise
    print('Loading data')
    T, V, test = load('/home/daniel/Code/NN/net/data/mnist.pkl.gz')
    start = time.time()
    print('Computing first layer')
    W = mDA(T[0].T)
    print('Time for first layer: %.4f' % (time.time()-start,))
    
    num_imgs = 400
    index = rp.choice(W.shape[1], num_imgs, replace=False)
    img = W[:,index]
    image = Image.fromarray(utils.tile_raster_images(X=img.T,
             img_shape=(28,28), tile_shape=(20, 20),
             tile_spacing=(1, 1)))
    image.save('hypSDA.png')
   
   
   