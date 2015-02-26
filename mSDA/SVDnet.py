'''
SVDnet.py
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


fmda = fmDA()
# Note the data is stored row-wise and the fmDA takes it column-wise
print('Loading data')
T, V, test  = fmda.load('../net/data/mnist.pkl.gz')
X           = np.vstack((T[0],V[0])).T
k           = 2
w           = (400, 200)
kappa       = 0.7

params      = fmda.SVDnet(X, k, w, kappa)