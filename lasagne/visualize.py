'''Visualize and analyze weights'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import numpy 


def visualize(file_name):
    '''Open up all the weights in the network'''
    file = open(file_name, 'r')
    data = cPickle.load(file)
    file.close()
    for arr in data:
        print arr.shape


if __name__ == '__main__':
    file = 'model.npz'
    visualize(file)
