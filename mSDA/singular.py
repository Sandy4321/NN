'''
singular.py is a script to calculate the singular value decomposition
of the single layer mDA on a few different data sets to motivate the
use of undercomplete layers for dimensionality reduction
'''
from fmDA import fmDA
import numpy as np
import numpy.random as rp
import time
import cPickle
import gzip
import utils
from PIL import Image
import sys
import matplotlib.pyplot as plt

fmda = fmDA()
# Note the data is stored row-wise and the fmDA takes it column-wise
print('Loading data')
T, V, test  = fmda.load('../net/data/mnist.pkl.gz')
X           = np.vstack((T[0],V[0])).T
Xtest       = test[0].T
# Setup test case
side_length = 20
num_imgs    = side_length**2
Xtest       = Xtest[:,:num_imgs]
mask        = rp.random_sample(Xtest.shape)
for cols in xrange(side_length):
    p       = (cols+1.)/side_length
    i       = cols*side_length
    mask[:,i:(i+side_length)]   = (mask[:,i:(i+side_length)] > p)*1.
Xtest       = Xtest*mask
image       = Image.fromarray(utils.tile_raster_images(X=Xtest.T, \
                                                   img_shape=(28,28), tile_shape=(20, 20), \
                                                   tile_spacing=(1, 1)))

# Train model
print('Training')
for i in xrange(10):
    params_SDA  = fmda.SDA('krfmDA',X,k=1,kappa=i*0.1)
    for param in params_SDA:
        U,S,V = np.linalg.svd(param[:,:-1])
        plt.plot(S)
    
plt.xlabel('Singular component')
plt.ylabel('Singular values')
plt.title('Singular spectrum of linear layer')
plt.yscale('log')
plt.grid()
plt.show()



