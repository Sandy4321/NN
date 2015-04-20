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


x = numpy.zeros((4,1))
print x
x_gpu = TsharedX(x, 'x_gpu', borrow=True)

srng = RandomStreams(seed=234)
y_gpu = srng.uniform((4,1))

fnc = Tfunction([], outputs=x_gpu+y_gpu)
print fnc()
print fnc()

z = numpy.zeros((4,5))
print z
z_gpu = TsharedX(z, 'z_gpu', borrow=True)

fnc2 = Tfunction([],
    outputs=x_gpu+y_gpu,
    givens={
        x_gpu : z_gpu
    })

print fnc2()
print fnc2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
