'''
Whitening, PCA-whitening and ZCA-whitening
'''

import numpy as np
from data_handling import Data_handling
import matplotlib.pyplot as plt
import utils
from PIL import Image


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
    Sx = np.cov(X)
    
    image = Image.fromarray(utils.tile_raster_images(X=Sx[0:10,0:10],
             img_shape=(11, 11), tile_shape=(1, 1),
             tile_spacing=(1, 1)))
    image.save('cov.png')

    print('done')
    
    
if __name__ == '__main__':
    whiten(2,3)
