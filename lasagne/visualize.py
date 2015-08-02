'''Visualize and analyze weights'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import numpy
import theano

from matplotlib import pyplot as plt

def visualize(file_name):
    '''Open up all the weights in the network'''
    file = open(file_name, 'r')
    data = cPickle.load(file)
    file.close()
    # M first layer
    M = data['Ml_hid1'][:784,:]
    R = data['Rl_hid1'][:784,:]
    S = numpy.log(1. + numpy.exp(R))
    M = numpy.reshape(M, (28,28,-1), order='F')
    S = numpy.reshape(S, (28,28,-1), order='F')
    n = 14
    P = numpy.zeros((28*n,28*n))
    k = 0
    for i in numpy.arange(n):
        for j in numpy.arange(n):
            F = M[:,:,k] + S[:,:,k] * numpy.random.randn(28,28)
            Plocal = F - numpy.amin(F) + 1e-4
            P[28*i:28*(i+1),28*j:28*(j+1)] = Plocal/numpy.amax(Plocal)
            k += 1
    fig = plt.figure()
    plt.imshow(P, interpolation='nearest', cmap='gray',
               vmin = numpy.amin(P), vmax = numpy.amax(P))
    print numpy.amin(P), numpy.amax(P)
    plt.show()


if __name__ == '__main__':
    file = 'model.npz'
    visualize(file)
