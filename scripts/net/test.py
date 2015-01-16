import numpy as np
import theano
import theano.tensor as T

a = T.fscalar('a')
#b = T.fscalar('b')

x = T.fscalar('x')
#y = T.fscalar('y')

b = 2*a
y = 3*x


c = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=theano.config.floatX)
c = theano.shared(c)


index = T.lscalar('index')

bfromc = theano.function([index], b, givens = {a: c[index]})
yfromx = theano.function([x], y)

print(bfromc(3), yfromx(10))

x = b
y = 3*x

yfromc = theano.function([index], y, givens = {a: c[index]})

print (yfromc(2))

