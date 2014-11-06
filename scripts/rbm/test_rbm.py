# test_rbm.py
#
# Daniel E. Worrall --- 27 Oct 2014
#
# Set of unit tests for the rbm extention script

# Test 1
# Do we generate samples from a bernoulli distribution roughly?

import numpy as np
import rbm_header as rbm
print "Testing vector sampler"

def test_rbm_smpl():
    '''Test vector sampler'''
    
    import numpy as np
    import rbm_header as rbm
    
    n = 10000.0
    K = np.linspace(1,2,10)
    M = [0.0, 0.1, 0.25,  0.5, 0.75, 1.0]
    
    fails = np.zeros((60))
    
    for i,m in enumerate(M):
        s = np.sqrt(m*(1-m))
        arg = np.abs(rbm.bern_samp(m * np.ones((n,1)),(n,1)) - m)
        for j,k in enumerate(K):
            P = sum(arg > k*s)/n
            if (P > (1/k**2)):
                fails[i*10 + j] = True
    
    print "Failed at ", [i for i,j in enumerate(fails) if j == True]
    assert sum(fails) == False


def test_rbm_smpl_mat():
    '''Test matrix sampler'''
    
    import numpy as np
    import rbm_header as rbm
    
    h = 1000
    w = 1000
    K = np.linspace(1,2,10)
    M = [0.0, 0.1, 0.25,  0.5, 0.75, 1.0]
    
    fails = np.zeros((60))
    
    for i,m in enumerate(M):
        s = np.sqrt(m*(1-m))
        arg = np.abs(rbm.bern_samp_mat(m * np.ones((h,w)),(h,w)) - m)
        for j,k in enumerate(K):
            P = sum(sum(arg > k*s))/(h*w)
            if (P > (1/k**2)):
                fails[i*10 + j] = True
    
    print "Failed at ", [i for i,j in enumerate(fails) if j == True]
    assert sum(fails) == False
    
    

































