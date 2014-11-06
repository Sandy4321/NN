# test_rbm.py
#
# Daniel E. Worrall --- 27 Oct 2014
#
# Set of unit tests for the rbm extention script

import numpy as np
import rbm_header as rbm

# Test 1
# Do we generate samples from a bernoulli distribution roughly?


print "Testing vector sampler"
n = 100000.0
m = [0.0, 0.1, 0.25,  0.5, 0.75, 1.0]
print "Mean:", m[0], "empirical mean:", np.average(rbm.bern_samp(m[0] * np.ones((n)),n))
print "Mean:", m[1], "empirical mean:", np.average(rbm.bern_samp(m[1] * np.ones((n)),n))
print "Mean:", m[2], "empirical mean:", np.average(rbm.bern_samp(m[2] * np.ones((n)),n))
print "Mean:", m[3], "empirical mean:", np.average(rbm.bern_samp(m[3] * np.ones((n)),n))
print "Mean:", m[4], "empirical mean:", np.average(rbm.bern_samp(m[4] * np.ones((n)),n))
print "Mean:", m[5], "empirical mean:", np.average(rbm.bern_samp(m[5] * np.ones((n)),n))

# Test 2
print "Testing matrix sampler"
R = 1000
K = 1000
m = [0.0, 0.1, 0.25,  0.5, 0.75, 1.0]
print "Mean:", m[0], "empirical mean:", np.average(rbm.bern_samp_mat(m[0] * np.ones((R,K)),(R,K)))
print "Mean:", m[1], "empirical mean:", np.average(rbm.bern_samp_mat(m[1] * np.ones((R,K)),(R,K)))
print "Mean:", m[2], "empirical mean:", np.average(rbm.bern_samp_mat(m[2] * np.ones((R,K)),(R,K)))
print "Mean:", m[3], "empirical mean:", np.average(rbm.bern_samp_mat(m[3] * np.ones((R,K)),(R,K)))
print "Mean:", m[4], "empirical mean:", np.average(rbm.bern_samp_mat(m[4] * np.ones((R,K)),(R,K)))
print "Mean:", m[5], "empirical mean:", np.average(rbm.bern_samp_mat(m[5] * np.ones((R,K)),(R,K)))

