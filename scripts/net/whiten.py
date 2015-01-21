'''
Whitening, PCA-whitening and ZCA-whitening
'''

import numpy as np
from data_handling import Data_handling
import matplotlib.pyplot as plt
import utils
from PIL import Image
from pylab import *

dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')

def whiten(variant, epsilon):
    ''' This script is to whiten the input data '''
    
    print('Whitening')
    # Collect data into single object X
    train = dh.train_set_x.get_value(borrow=True)
    valid = dh.valid_set_x.get_value(borrow=True)
    test = dh.test_set_x.get_value(borrow=True)
    X = np.vstack((train, valid, test))
    
    mx = np.mean(X, axis=1)[:,np.newaxis]
    X -= mx   # relying on broadcasting here
  
    Sx = np.dot(X.T,X)/N
    print Sx.shape
    imsave('cov.png', Sx, cmap=cm.binary)
    grid(True)

    
    
if __name__ == '__main__':
    whiten(2,3)
