# RBM sampler
#
# Daniel E. Worrall --- 27 Oct 2014
#
# Generate samples from a trained RBM
import numpy as np

f = open("../../data/params.data")
lines=f.readlines()
b = lines[1]
print b
W = np.loadtxt("../../data/params.data", skiprows=3)