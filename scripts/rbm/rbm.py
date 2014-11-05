# RBM example
#
# Daniel E. Worrall --- 23 Oct 2014
#
# Use the contrastive divergence method to train an RBM with Gaussian inputs
# and Bernoulli hidden units --- note input must be square!!!

# Imports
import numpy as np                         # for matrix operations
import matplotlib.pyplot as plt            # for weight visualisation
import rbm_header as rbm                   # rbm_specific functions

# Parameters
alpha = np.exp(-1)             # learning rate numerator
K = 250                 # num hidden units
batch_size = 10.0       # mini-batch size
max_epochs = 10        # number of epochs until finished
t = 1                  # time between cmd line progress updates
beta = 0.0005           # weight-cost (regularisation)
rho = 0.05              # sparsity target
l = 0.95                # sparsity decay
PCD_size = 10           # number of fantasy particles
inFile = "../../data/mldata/mnist_bin.data"
outFile = "../../data/rbm/params_mnist.data"
outFile2 = "../../data/rbm/params_stats_mnist.data"

# Load data
print "Loading data"
v = np.loadtxt(inFile, delimiter = ',')
(N,R) = v.shape
print "Number of training samples =",N
print "Number of dimensions =",R


# Calculate/initialise some data-dependent parameters for training. We use
# the intialisation of weights and biases as per Hinton (2010) practical
# recommendations noting that the biases are initialised low for sparsity.
# May change this....
n_batches = np.ceil(N/batch_size)
W = 0.01*np.random.randn(R,K)
b = 0.01*np.random.randn(R)
c = 0.01*np.random.randn(K)-4
W_size = np.zeros((max_epochs,1))
q = np.zeros((K))             # tracked estimate of the mean hidden activation probability
F = 0.01*np.random.randn(R,PCD_size)

# Training loop
for epoch in np.arange(max_epochs):
    alpha_t = alpha*(max_epochs-epoch)/max_epochs
    for B in np.arange(n_batches):
       
        gW = np.zeros((R,K))    # gradient of weights
        gb = np.zeros((R))   # visible bias gradient
        gc = np.zeros((K))    # hidden bias gradient
        
        for i in np.arange(B*batch_size,min((B+1)*batch_size,N)):
                      
            # Calculate the expectation over the model distribtution using PCD
            vi = v[i,:]
           
            # Perform the Persistent CD algorithm
            (Eh, h2, F) = rbm.PCD(b,c,W,F,vi,PCD_size,K,R)
            
            # Update cumulative gradients and mean activation estimate
            gW += np.outer(vi,Eh) - (np.einsum('ik,jk',F,h2)/PCD_size) # efficient way to evaluate sum of outer products
            gb += vi - np.average(F,axis=1)
            gc += Eh - np.average(h2,axis=1)
            
            q = l*q + (1-l)*Eh
    
        # Update weights and biases, note the weight decay term
        batch_size2 = min((B+1)*batch_size,N) - B*batch_size
        
        W += alpha_t*(gW/batch_size2 - beta*rbm.sparsity(q,rho))
        b += alpha_t*gb/batch_size2 
        c += alpha_t*(gc/batch_size2 - beta*rbm.sparsity(q,rho))
        
        W_size[epoch] = np.linalg.norm(W,'fro')
    if (epoch%t == 0):    
        print "Iteration: %d \t |W|_F: %.3f \t |b|_F: %.3f \t |c|_F: %.3f" \
            % (epoch, W_size[epoch], np.linalg.norm(b), np.linalg.norm(c))

# Save data to file
rbm.save((W_size,b,c,W),outFile,outFile2)

# Visualise weights as a grid

print "Loading weights for visualisation"
sqrtK = np.ceil(np.sqrt(K))
for k in np.arange(K):
    plt.subplot(sqrtK,sqrtK,k)
    img = W[:,k].reshape((np.sqrt(R),np.sqrt(R)))
    plt.imshow(img, cmap=plt.cm.gray,interpolation='nearest')
    plt.axis('off')
# Show
plt.show()



































