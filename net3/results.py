'''Plot results of dropout mixture model'''
import fnmatch
import os

import cPickle
import numpy


for file in os.listdir('./pkl/'):
    if fnmatch.fnmatch(file, '*best_dropprior*'):
        stream = open(file, 'r')
        state = cPickle.load(stream)
        stream.close()
        
        monitor = state['monitor']
        args = state['hyperparams']
        best_cost = monitor['best_cost']

        params = monitor['best_model']


