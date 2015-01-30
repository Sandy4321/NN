from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams

N = 100
iters = 1000
rng = numpy.random.RandomState(22)
R = theano.shared(np.asarray(rng.rand(N,N), dtype=theano.config.floatX))




theano_rng = RandomStreams(seed=32)
rn = theano_rng.binomial(size=(N,N), n=1, p=0.5).astype(theano.config.floatX)
f = function([], rn)
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)




theano_rng = RandomStreams(seed=32)
rn = theano_rng.binomial(size=(N,N), n=1, p=0.5).astype(theano.config.floatX)
f = function([], sandbox.cuda.basic_ops.gpu_from_host(rn))
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)



rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(N,N), config.floatX))
f = function([], x)
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)


rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(N,N), config.floatX))
f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)



rng = numpy.random.RandomState(22)
t0 = time.time()
for i in xrange(iters):
    r = rng.rand(N,N)
t1 = time.time()

print(t1-t0)




rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(N,N), config.floatX))
f = function([], sandbox.cuda.basic_ops.gpu_from_host(x*R))
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)


rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(N,N), config.floatX))
f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.mul(x,R)))
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)



rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.randint(0,2,(N,N)), config.floatX))
f = function([], x)
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print(t1-t0)
print(x.get_value())


theano_rng
t0 = time.time()
for i in xrange(iters):
    r = theano_rng.uniform(size=(N,N))
t1 = time.time()

print(t1-t0)



























