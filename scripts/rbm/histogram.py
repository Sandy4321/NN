# RBM histogram
#
# Daniel E. Worrall --- 29 Oct 2014
#
# Visualise key statistics of the trained RBM

# Imports
import numpy as np
import matplotlib.pyplot as plt
import rbm_header as rbm

# Load data
f1 = "../../data/params_mnist.data"
f2 = "../../data/params_stats_mnist.data"

print "Loading data"
f = open(f1)
# Extract and reshape RBM parameters
for i, line in enumerate(f):
    if i == 0:
        b = np.fromstring(line, dtype=float, sep=' ')
        b = b[np.newaxis].T
        h = len(b)
    if i == 1:
        c = np.fromstring(line, dtype=float, sep=' ')
        c = c[np.newaxis].T
        w = len(c)
        W = np.zeros((h,w))     # I know this is terrible practise, but it
    if i > 1:
        W[i-2,:] = np.fromstring(line, dtype=float, sep=' ')
        
f.close()

print "Loading stats"
f = open(f2)
for i, line in enumerate(f):
    if i == 0:
        W_size = np.fromstring(line, dtype=float, sep=' ')
        W_size = W_size[np.newaxis].T


# Display graph of frobenius norm of weights over training
print "1/3 Graph of frobenius norm of weights"
fig = plt.figure(1)
plt.plot(W_size)
plt.title('Frobenius norm of weights')
plt.xlabel('Iteration number')
plt.ylabel('Norm of weights')
plt.grid()
plt.show()


# Display histogram of the weights
print "2/3 Histogram of the final weight matrix"
print "max = %.8f, min = %.8f" % (W.max(), W.min())
n, bins, patches = plt.hist(np.hstack(W), bins=100, range=(-0.5,0.5), normed=True,
                            weights=None, cumulative=False, facecolor='green', alpha=0.75)

plt.xlabel('Weights')
plt.ylabel('Probability')
plt.title('Clipped histogram of weights')
plt.grid()

plt.show()

# Display histogram of the weights
# create a new data-set
print "3/3 Histogram of the biases"
n, bins, patches = plt.hist([b,c], 25, normed=1, histtype='bar',
                            color=['crimson','chartreuse'],
                            label=['visible','hidden'])
plt.legend()
plt.show()








































