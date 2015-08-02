'''Visualize and analyze weights'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy 


class Visualize(object):
    def visualize(self, file):
        '''Open up all the weights in the network'''
        data = numpy.load(file)
        print data['arr_0'].shape


if __name__ == '__main__':
    file = 'model.npz'
    Visualize().visualize(file)
