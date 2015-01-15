import numpy as np
import theano
import theano.tensor as T

a = T.lscalar('a')
b = T.lscalar('b')

x = T.lscalar('x')
y = T.lscalar('y')

b = 2*a
y = 3*x

c = theano.shared(3)


bfroma = theano.function([], b, givens = {a: c})
yfromx = theano.function([x], y)

print(bfroma(), yfromx(4))

#x = b

#y = 3*b

yfroma = theano.function([], [y], givens = {x : b, a : c})

print yfroma()

'''
So the synopsis is that you have to rebuild the computational graph when you
rebind an input
'''