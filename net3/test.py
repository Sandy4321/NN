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

from theano import sparse

X = theano.tensor.matrix('x')
W = theano.tensor.matrix('W')
B = theano.tensor.matrix('B')
H = theano.tensor.dot(B*W,X)
loss = theano.tensor.sum(H**2)

b = numpy.random.rand(5,40) < 0.5
w = numpy.arange(200).reshape(5,40).astype(theano.tensor.config.floatX)
x = numpy.arange(200).reshape(40,5).astype(theano.tensor.config.floatX)

Tfunc = theano.function([B,W,X],H)
print Tfunc(b,w,x)

grad = theano.gradient.grad(loss,W)
Tgrad = theano.function([B,W,X],grad)
print Tgrad(b,w,x).shape

S = sparse.csc_from_dense(B*W)
Hs = sparse.basic.dot(S,X)
losss = theano.tensor.sum(Hs**2)

Tfuncs = theano.function([B,W,X],Hs)
print Tfuncs(b,w,x)

grads = theano.gradient.grad(losss,S)
Tgrads = theano.function([B,W,X],grads)
print Tgrads(b,w,x).shape

'''
data = numpy.load('pkl/pruned.pkl.npz')
x = numpy.logspace(-5.,-3.,num=35)

kl = data['kl'] #.mean(axis=1)
snr = data['snr'] #.mean(axis=1)
plt.figure()
plt.loglog(x,kl,'r')
plt.loglog(x,snr,'b')
plt.show()
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
