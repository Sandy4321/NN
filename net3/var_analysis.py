'''Variance analysis'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import numpy
import theano

from DGWN import Dgwn
from matplotlib import pyplot as plt


args = {
    'mode' : 'training',
    'layer_sizes' : (100, 100, 100),
    'nonlinearities' : ('linear', 'linear'),
    'prior': 'Gaussian',
    'prior_variance' : 1/200.,
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
N = 1
T = 100
x = theano.tensor.matrix('x')
for j in numpy.arange(3):
    args['prior_variance'] = 1./10 # (1/1.)/(10.**j)
    dgwn = Dgwn(args)
    output, _ = dgwn.predict(x,args)
    Tfunc = theano.function(inputs=[x], outputs=output)
    Tgrad = theano.function(inputs=[x], outputs=gradient(output,x))
    Tcost = theano.function(inputs=[x], outputs=grad(output[0,0],x))
    y = numpy.zeros((100,N*T))
    dy = numpy.zeros((100,N*T))
    cost = numpy.zeros((100,N*T))
    for i in numpy.arange(T):
        input = numpy.sqrt(k*(10**j))*numpy.random.randn(100,N).astype(theano.config.floatX)
        y[:,i*N:(i+1)*N]  = Tfunc(input)
        dy[:,i*N:(i+1)*N] = Tgrad(input)
        cost[:,i*N:(i+1)*N]= Tcost(input)
    
    print numpy.var(y)
    print numpy.var(dy)/(4.*numpy.var(y))
    print numpy.var(cost)



'''
J = 100
M = 1000
x = numpy.zeros(J)

for i in numpy.arange(J):
    #z = numpy.random.randn(i+1,M)
    eps = numpy.random.randn(i+1,M)
    x[i] = numpy.mean(numpy.sum(eps**2,axis=0))
    #dz = numpy.sum(eps*z,axis=0)/numpy.sqrt(numpy.sum(z**2,axis=0))
    #x[i] = numpy.var(dz)
   
fig = plt.figure()
plt.loglog(numpy.arange(J),x,'b')
plt.show()
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
