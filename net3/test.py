"""Autoencoder exhibiting the dropprior regulariser"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import sys, time

import cPickle
import gzip
import numpy
import theano.tensor as T
import theano.tensor.nnet as Tnet

from matplotlib import pyplot as plt
from theano import config as Tconf
from theano import function as Tfunction
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams


def add(a,b):
    return T.sum(a,axis=1,keepdims=True) + b 

x = numpy.ones((4,2)).astype(Tconf.floatX)
x_gpu = TsharedX(x, 'x_gpu', borrow=True)

y = numpy.ones((4,1)).astype(Tconf.floatX)
y_gpu = TsharedX(y, 'y_gpu', borrow=True)

add_result = add(x_gpu, y_gpu)

fnc = Tfunction([], outputs=add_result)

print fnc()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
