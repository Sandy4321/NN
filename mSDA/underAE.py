'''
kfmDA.py
'''
from kfmDA import kfmDA
import numpy as np
import numpy.random as rp
import time
import cPickle
import gzip
import utils
from PIL import Image
import sys
import matplotlib.pyplot as plt

kfmda = kfmDA()
# Note the data is stored row-wise and the kfmDA takes it column-wise
print('Loading data')
T, V, test  = kfmda.load('../net/data/mnist.pkl.gz')
X           = np.vstack((T[0],V[0])).T
Xtest       = test[0].T
# Setup test case
side_length = 20
num_imgs    = side_length**2
Xclean      = Xtest[:,:num_imgs]
mask        = rp.random_sample(Xclean.shape)
for cols in xrange(side_length):
    p       = (cols+1.)/side_length
    i       = cols*side_length
    mask[:,i:(i+side_length)]   = (mask[:,i:(i+side_length)] > p)*1.
Xdirty      = Xclean*mask


# Preprocess data
print('Training')
meanX   = np.mean(X,axis=1)[:,np.newaxis]
X0      = X - meanX

# Train
params_SDA  = kfmda.SDA('underAE',X0,H=(100,50))
Y           = kfmda.map(Xclean, params_SDA)
error   = ((Y-Xclean)**2).sum()
print('Error: %0.3g' % (error,))

# Test images
image       = Image.fromarray(utils.tile_raster_images(X=Y.T, \
                                                   img_shape=(28,28), tile_shape=(20, 20), \
                                                   tile_spacing=(1, 1)))
image.save('kfmDA.png')





































