import numpy as np
import numpy.random as rand
import theano
import theano.tensor as T

'''
p=0.2

# Generate out data matrix
M = 100
N = 102
D = rand.randn(N,N)
v = rand.randn(N)

Q = 100000
S = 0
# Generate our random mask
for i in xrange(Q):
    R = rand.random_sample(v.shape)
    R = (R>(1-p))*1
    R2 = rand.random_sample(v.shape)
    R2 = (R2>(1-p))*1
    S += np.dot(R*v,np.dot(D,R*v))
    
S/=Q
print S/np.dot(v,np.dot(D,v))
'''

def salt_and_pepper(IN, p = 0.2):
        # salt and pepper noise
        print 'DAE uses salt and pepper noise'
        a = (rand.random_sample(IN.shape)>(1-p))*1
        b = (rand.random_sample(IN.shape)>0.5)*1
        c = ((a==0) * b)
        print a
        print b
        print c
        print IN * a + c
        
if __name__ == '__main__':
    IN = 0.5*np.ones((5,5))
    salt_and_pepper(IN)