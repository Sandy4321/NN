'''pyGP'''

import numpy
import pyGPs

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = numpy.linspace(0,6,10)
y = numpy.linspace(0,6,10)
x, y = numpy.meshgrid(x,y)
coords = numpy.vstack((x.flatten(), y.flatten())).T
z = numpy.sin(coords[:,0] + 0.5*coords[:,1]) + 0.2*numpy.random.randn(100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:,0], coords[:,1], z)
plt.show()

x = numpy.linspace(0,6,50)
y = numpy.linspace(0,6,50)
new_x, new_y = numpy.meshgrid(x,y)
new_coords = numpy.vstack((new_x.flatten(), new_y.flatten())).T

model = pyGPs.GPR()      # specify model (GP regression)
model.getPosterior(coords, z) # fit default model (mean zero & rbf kernel) with data
model.optimize(coords, z)     # optimize hyperparamters (default optimizer: single run minimize)
ym, ys2, fm, fs2, lp = model.predict(new_coords)         # predict test cases

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:,0], coords[:,1], z, c='r')
ax.scatter(new_coords[:,0], new_coords[:,1], fm)
plt.show()
