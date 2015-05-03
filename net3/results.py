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

mesh = 100
m = numpy.linspace(0.,1,mesh)
s = numpy.linspace(0.,1,mesh)/numpy.sqrt(12)
new_x, new_y = numpy.meshgrid(m,s)
new_coords = numpy.vstack((new_x.flatten(), new_y.flatten())).T

model = pyGPs.GPR()      # specify model (GP regression)
model.getPosterior(coords, acc) # fit default model (mean zero & rbf kernel) with data
model.optimize(coords, acc)     # optimize hyperparamters (default optimizer: single run minimize)
ym, ys2, fm, fs2, lp = model.predict(new_coords)         # predict test cases
fm = numpy.reshape(fm, (mesh, mesh))

fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(coords[:,0], coords[:,1], acc, c='r')
#ax.plot_surface(new_x, new_y, fm, alpha=0.3)
plt.imshow(fm,cmap=cm.RdBu) # drawing the function
cset = plt.contour(fm.T,numpy.arange(1.5,3.5,0.2),linewidths=2,cmap=cm.Set2)
plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#new_z = numpy.reshape(fm, new_x.shape)
#CS = plt.contour(new_x, new_y, new_z)
#plt.clabel(CS, CS.levels, inline=1, fontsize=10)
plt.show()


mesh = 100
for p in numpy.linspace(0.5,0.5,1):
    m = numpy.linspace(p,p,1)
    s = numpy.linspace(0.,1,mesh)/numpy.sqrt(12)
    new_x, new_y = numpy.meshgrid(m,s)
    new_coords = numpy.vstack((new_x.flatten(), new_y.flatten())).T
    ym, ys2, fm, fs2, lp = model.predict(new_coords)         # predict test cases
    nc = new_coords[:,1]
    plt.plot(nc, fm)
    sd = numpy.vstack((fm+2*numpy.sqrt(fs2),fm[::-1]-2*numpy.sqrt(fs2[::-1])))
    coords = numpy.hstack((nc,nc[::-1]))
    plt.fill(coords, sd, alpha=0.1)
    plt.xlim(0,0.3)
plt.show()
































