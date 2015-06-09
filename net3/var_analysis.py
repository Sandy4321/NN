'''Variance analysis'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import numpy
import scipy.special as ss
import theano

from DGWN import Dgwn
from matplotlib import pyplot as plt


N = 1
T = 100
args = {
    'mode' : 'training',
    'layer_sizes' : (100, 100, 100, 100, 100),
    'nonlinearities' : ('linear','linear', 'linear', 'linear'),
    'prior_variance' : 1e-3,
    'num_components' : 1,
    'num_samples' : 1,
    'dropout_dict' : None,
    'cov' : False,
    }

def gradient(output, wrt):
    cost = costfn(output)
    return grad(cost, wrt)

def costfn(output):
    return theano.tensor.sum(output**2)

def grad(cost, wrt):
    return theano.grad(cost, wrt)

k = 1.
x = theano.tensor.matrix('x')
for j in numpy.arange(3):
    args['prior_variance'] = 1/100.
    dgwn = Dgwn(args)
    output, _ = dgwn.predict(x,args)
    Tfunc = theano.function(inputs=[x], outputs=output)
    Tgrad = theano.function(inputs=[x], outputs=gradient(output,x))
    Tcost = theano.function(inputs=[x], outputs=grad(output[0,0],x))
    y = numpy.zeros((100,N*T))
    dy = numpy.zeros((100,N*T))
    cost = numpy.zeros((100,N*T))
    for i in numpy.arange(T):
        input = numpy.sqrt(k)*numpy.random.randn(100,N).astype(theano.config.floatX)
        y[:,i*N:(i+1)*N]  = Tfunc(input)
        dy[:,i*N:(i+1)*N] = Tgrad(input)
        cost[:,i*N:(i+1)*N]= Tcost(input)
    
    print numpy.var(y)
    print numpy.var(dy)
    print numpy.var(cost)




'''
J = 100
N = 100
M = 1000
x = numpy.zeros(J)

for i in numpy.arange(J):
    z = numpy.random.randn(i+1,M)
    eps = numpy.random.randn(N)
    dz = numpy.outer(eps,z[:,:]/numpy.sqrt(numpy.sum(z**2,axis=0)))
    x[i] = numpy.var(dz)
   
fig = plt.figure()
plt.loglog(numpy.arange(J),x,'b')
plt.loglog(numpy.arange(J),1./numpy.arange(J),'r')
plt.show()
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
