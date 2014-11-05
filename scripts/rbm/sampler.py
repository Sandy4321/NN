# RBM sampler
#
# Daniel E. Worrall --- 29 Oct 2014
#
# Visualise fantasies from a trained RBM

# Imports
import numpy as np
import matplotlib.pyplot as plt
import rbm_header as rbm

# Parameters
n = 1
n_smpl = 49    # Number of samples split n iterations apart

# Load data
f1 = "../../data/rbm/params_mnist.data"
f2 = "../../data/mldata/mnist_bin.data"

print "Loading data"
f = open(f1)

# Extract and reshape RBM parameters
for i, line in enumerate(f):
    if i == 0:
        b = np.fromstring(line, dtype=float, sep=' ')
        h = len(b)
    if i == 1:
        c = np.fromstring(line, dtype=float, sep=' ')
        w = len(c)
        W = np.zeros((h,w))     # I know this is terrible practise, but it
    if i > 1:
        W[i-2,:] = np.fromstring(line, dtype=int, sep=' ')
        
f.close()

# Now choose a number to fantasise. To begin with let's choose one at random
# To do this we shall choose a number from MNIST and feed that into the RBM as
# a seed and then block Gibbs sample for several iterations. The samples should
# eventually took number-like.
f = open(f2)

# Find the size of the file. Unfortuantely this involves initially iterating through
# the file until we find the end.
for i, line in enumerate(f):
    pass
N = i + 1
seed = np.random.choice(N, 1)

# Read off the image of the random number
f.seek(0)
for i, line in enumerate(f):
    if i == seed:
        v = np.fromstring(line, dtype=int, sep=',')

# Display the seed, so we have an idea of what the sampler is aiming for
R = len(v)
img = v.reshape((np.sqrt(R),np.sqrt(R)))
plt.imshow(img, cmap=plt.cm.gray,interpolation='nearest')
plt.axis('off')
plt.show()

# Now to do some sampling. We do this via the rbm_header method sample, which takes
# in all the RBM parameters, with some parameter sizes to make things a bit faster
# and a number n, which is the number of Gibbs iterations to perform before returning
# a fantasy.

print "Loading weights for visualisation"

sqrtK = np.ceil(np.sqrt(n_smpl))
for k in np.arange(n_smpl):
    v = rbm.sample(b,c,W,v,n,w,R)
    plt.subplot(sqrtK,sqrtK,k)
    img = v.reshape((np.sqrt(R),np.sqrt(R)))
    plt.imshow(img, cmap=plt.cm.gray,interpolation='nearest')
    plt.axis('off')
# Show
plt.show()



































