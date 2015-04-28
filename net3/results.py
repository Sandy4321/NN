'''Plot results of dropout mixture model'''
import fnmatch
import os

import cPickle
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
    if fnmatch.fnmatch(file, '*best_dropprior*'):
        stream = open(dir + file, 'r')
        state = cPickle.load(stream)
        stream.close()
        
        monitor = state['monitor']
        hyperparams = state['hyperparams']
        
        best_cost.append(monitor['best_cost'])
        lower = hyperparams['lower']
        upper = hyperparams['upper']
        
        params['mu'].append((lower+upper)/2.)
        params['std'].append((upper-lower)/numpy.sqrt(12))

mu = numpy.asarray(params['mu'])
std = numpy.asarray(params['std'])
acc = numpy.asarray(best_cost)

acc = (1 - acc)*100

coords = numpy.vstack((mu,std)).T

mesh = 12
m = numpy.linspace(0.,1,mesh)
s = numpy.linspace(0.,1,mesh)/numpy.sqrt(12)
new_x, new_y = numpy.meshgrid(m,s)
new_coords = numpy.vstack((new_x.flatten(), new_y.flatten())).T

model = pyGPs.GPR()      # specify model (GP regression)
model.getPosterior(coords, acc) # fit default model (mean zero & rbf kernel) with data
model.optimize(coords, acc)     # optimize hyperparamters (default optimizer: single run minimize)
ym, ys2, fm, fs2, lp = model.predict(new_coords)         # predict test cases

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(coords[:,0], coords[:,1], acc, c='r')
ax.scatter(new_coords[:,0][:,numpy.newaxis], new_coords[:,1][:,numpy.newaxis], fm, c='b')
#new_z = numpy.reshape(fm, new_x.shape)
#CS = plt.contour(new_x, new_y, new_z)
#plt.clabel(CS, CS.levels, inline=1, fontsize=10)
plt.show()



































