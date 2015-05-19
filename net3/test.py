"""Autoencoder exhibiting the dropprior regulariser"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import sys, time

import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy
import scipy.special as spsp
import theano

from draw_beta import Draw_beta


a = theano.tensor.scalar('a')
b = theano.tensor.scalar('b')
u = theano.tensor.matrix('u')

#d_a = 2*numpy.random.random_sample((1000,1)).astype(theano.config.floatX) + 1.
#d_b = 2*numpy.random.random_sample((1000,1)).astype(theano.config.floatX) + 1.
d_a = 5.
d_b = 2.
d_u = numpy.random.random_sample((1000,1)).astype(theano.config.floatX)

p = Draw_beta()(a, b, u)

Tfunc = theano.function([a, b, u], p)

q = Tfunc(d_a, d_b, d_u)
print q
plt.hist(q, 50, range=[0.,1.], normed=1)
plt.show()












    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
