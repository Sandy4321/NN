'''
Whitening, PCA-whitening and ZCA-whitening
'''
import theano
import numpy as np
import scipy as sp
from data_handling import Data_handling
import utils
from PIL import Image
import pickle
import matplotlib.pyplot as plt

dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')

def whiten(epsilon, file_name):
    ''' This script is to whiten the input data '''
    
    print('Whitening')
    # Collect data into single object X
    train = dh.train_set_x.get_value(borrow=True)
    valid = dh.valid_set_x.get_value(borrow=True)
    test = dh.test_set_x.get_value(borrow=True)
    
    ltrain = train.shape[0]
    lvalid = valid.shape[0] + ltrain
    ltest = test.shape[0] + lvalid
    X = np.vstack((train, valid, test))
    
    # Zero mean
    mx = np.mean(X, axis=1)[:,np.newaxis]
    X -= mx   # relying on broadcasting here
    # Covariance decomposition
    Sx = np.dot(X.T,X)/X.shape[0]
    U,S,V = np.linalg.svd(Sx)       # note that S is a (n,) diag
    # Robust ZCA
    Xrot = np.dot(U.T,X.T)
    Sinv = np.diag(1.0/np.sqrt(S + epsilon))
    Wrob = np.dot(Sinv,Xrot)
    #print Srob
    Z = np.dot(U,Wrob).T
    
    # Here we ZCA some corrupted test images
    dh.get_corrupt(corruption_level=0.2)
    C = dh.corrupt_set_x.get_value(borrow=True)
    Crot = np.dot(U.T,C.T)
    CWrob = np.dot(Sinv,Crot)
    CZ = np.dot(U,CWrob).T
    CZ = CZ.astype(theano.config.floatX)
    
    image = Image.fromarray(utils.tile_raster_images(X=Z,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
    image.save('ZCA.png')

    image = Image.fromarray(utils.tile_raster_images(X=CZ,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
    image.save('ZCA_corrupt.png')

    
    print('Pickling')
    train = Z[0:ltrain,:].astype(theano.config.floatX)
    valid = Z[ltrain:lvalid,:].astype(theano.config.floatX)
    test = Z[lvalid:ltest,:].astype(theano.config.floatX)
    dh.train_set_x.set_value(train, borrow=True)
    dh.valid_set_x.set_value(valid, borrow=True)
    dh.test_set_x.set_value(test, borrow=True)
    dh.corrupt_set_x.set_value(CZ, borrow=True)
    stream = open(file_name,'w')
    pickle.dump(dh, stream)
    stream.close()
    
if __name__ == '__main__':
    whiten(0.001,'ZCA_data.pkl')










































