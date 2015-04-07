"""Undercomplete fmDA implementation"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import numpy

from fmDA import fmDA

fmda = fmDA()
# Note the data is stored row-wise and the kfmDA takes it column-wise
print('Loading data')
train, valid, test = fmda.load('../net/data/mnist.pkl.gz')
X = train[0].astype(dtype=numpy.float32)

# Train
print('Training')
#params_SDA = kfmda.SDA('underAE',X0,H=(200,))
B, bE, bD = fmda.underfmDA(X, X, H=200)



































