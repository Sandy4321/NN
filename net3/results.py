'''Plot results of dropout mixture model'''
import fnmatch
import os

import cPickle
import matplotlib.cm as cm
import numpy
import pyGPs

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

i = 0
# Best costs is a 1D array
best_cost = []
# Params is a 2D array
params = {'mu' : [], 'std' : []}
dir = '/home/daniel/Code/NN/net3/pkl/'
for file in os.listdir(dir):
    if fnmatch.fnmatch(file, 'novar*'):
        if 'nag' in file:
            stream = open(dir + file, 'r')
            state = cPickle.load(stream)
            stream.close()
            
            monitor = state['monitor']
                    
            best_cost.append(monitor['best_cost'])
            
print numpy.mean(best_cost)
print numpy.std(best_cost)
print best_cost

































