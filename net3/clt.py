'''clt.py'''
import numpy

from matplotlib import pyplot as plt
from scipy.stats import norm

N = 1000
M = 80000
offset = numpy.random.rand(N,1)
x = numpy.random.laplace(loc=0.,scale=1.0,size=(N,M)) + offset 
X = x.sum(axis=0)
m = numpy.mean(X)
s = numpy.std(X)
d = numpy.linspace(numpy.amin(X), numpy.amax(X), num=200)
y = norm.pdf((d-m)/s)
hist, bin_edges = numpy.histogram(X, bins=100, density=True)
fig = plt.figure()
plt.semilogy(bin_edges[1:], hist, 'b')
plt.semilogy(d, y, 'r')
plt.scatter(numpy.sum(offset), 1., s= 15)
plt.show()