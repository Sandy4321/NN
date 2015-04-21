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


def add(a,b,p):
    return a + b < p

x = numpy.zeros((4,2)).astype(Tconf.floatX)
x_gpu = TsharedX(x, 'x_gpu', borrow=True)

batch_size = 2
length = 5

smrg = MRG_RandomStreams(seed=12345)
m_gpu = smrg.uniform(size=x_gpu.shape)
p = numpy.asarray([0.,0.2,0.8,1.0], dtype=Tconf.floatX)[:,numpy.newaxis]
p_gpu = TsharedX(p, 'p', broadcastable=(False,True))

add_result = add(x_gpu, m_gpu, p)

z = numpy.zeros((4,length*batch_size + 1)).astype(Tconf.floatX)
z_gpu = TsharedX(z, 'z_gpu', borrow=True)

index = T.iscalar(name='index')
fnc = Tfunction([index],
    outputs=add_result,
    givens={
        x_gpu : z_gpu[:,index*batch_size:(index+1)*batch_size]
    })


start = time.time()
for j in numpy.arange(length+1):
    print fnc(j)

time1 = time.time() - start

print time1

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
