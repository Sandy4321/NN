import theano
import theano.tensor as T

m = T.scalar('m')
S = T.scalar('S')
#x = m + S
y = m + S**2
x = y - S**2 + S
grad = theano.grad(x, y)
compute = theano.function([m,S], grad)

print compute(2, 3)