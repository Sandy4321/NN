"""Undercomplete fmDA implementation"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import numpy
import utils

from fmDA import fmDA
from PIL import Image

fmda = fmDA()
# Note the data is stored row-wise and the fmDA takes it column-wise
print('Loading data')
train, valid, test = fmda.load('../net/data/mnist.pkl.gz')
train_X = train[0].T
test_X = test[0].T

# Train
print('Training')
B, bE, bD = fmda.underfmDA(train_X, train_X, H=200)

# Reconstruction
R = fmda.reconstruct(test_X, B, bE, bD)
error = (((test_X - R)**2).sum())/test_X.shape[1]
print('Reconstruction error: %f' % (error,))


































