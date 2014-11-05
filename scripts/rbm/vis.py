# RBM example
#
# Daniel E. Worrall --- 23 Oct 2014
#
# Visualise the weights and some of the training data

# Imports
import numpy as np
import matplotlib.pyplot as plt
import rbm_header as rbm

# Parameters
K = 250    # num hidden units

f1 = "../../data/mldata/mnist_bin.data"
f2 = "../../data/rbm/params_mnist.data"

# Load data --- currently I've only considered my comma and space separated format.
# Loading the data takes some time, so I only read from my subcollection of lines.
f = open(f1)

# We also need to know the shape of the data file. Unfortunately, we have to read the
# entire file to know this.
for i, line in enumerate(f):
    if i == 0:
        R = len(np.fromstring(line, dtype=int, sep=','))
    pass
N = i + 1
f.seek(0)
print "Number of training samples =", N ,"Number of dimensions =",R

# Visualise a random subcollection of the training images as a root K by root K grid.
# We have placed the constraint that the data has to be square. Will need to change this.
samples =  np.random.choice(N, K, replace=False)
v = np.zeros((K,R))
j=0
for i, line in enumerate(f):
    if i in samples:
        v[j,:] = np.fromstring(line, dtype=int, sep=',')
        j+=1

sqrtK = np.ceil(np.sqrt(K))

print "Loading data"
for k in np.arange(K):
    plt.subplot(sqrtK,sqrtK,k)
    img = v[k,:].reshape((np.sqrt(R),np.sqrt(R)))
    plt.imshow(img, cmap=plt.cm.gray,interpolation='nearest')
    plt.axis('off')
    
# Show
print "Showing data"
plt.show()
f.close()


# Now we visualise th weights of the trained model. 
print "Loading weights"
W = np.loadtxt(f2, skiprows=2)
(R,K) = W.shape

# Visualise weights as a grid
sqrtK = np.ceil(np.sqrt(K))

for k in np.arange(K):
    plt.subplot(sqrtK,sqrtK,k)
    img = W[:,k].reshape((np.sqrt(R),np.sqrt(R)))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')
    
# Show
print "Showing weights"
plt.show()




































