# RBM example
#
# Daniel E. Worrall --- 23 Oct 2014
#
# Use the contrastive divergence method to train an RBM with Gaussian inputs
# and Bernoulli hidden units --- note input must be square!!!

# Imports
import numpy as np                         # for matrix operations
import matplotlib.pyplot as plt            # for weight visualisation
from numbapro import cuda
from numbapro.cudalib import curand        # fast random numbers
from numba import *
import os
from timeit import default_timer as timer
import math

class Rbm:
    
    def __init__(self,parameters):
 
        # Initialise parameters. I want to include a nice wrapper later so that we can
        # nice and self-contained
        print("Training parameters loading...")
        self.alpha = np.exp(-1)             # learning rate numerator
        self.K = 250                        # num hidden units
        self.batch_size = 10.0              # mini-batch size
        self.max_epochs = 10                # number of epochs until finished
        self.t = 1                          # time between cmd line progress updates
        self.beta = 0.0005                  # weight-cost (regularisation)
        self.rho = 0.05                     # sparsity target
        self.l = 0.95                       # sparsity decay
        self.PCD_size = 10                  # number of fantasy particles
        self.inFile = "../../data/preproc.npz"
        self.outFile = "../../data/params_mnist.npz"
        self.outFile2 = "../../data/params_stats_mnist.npz"
       
       
        # Load data
        self.load_data()
        
        
        # Hack for analysis only
        print("Reshuffling data...")
        self.train = self.train[:1000,:]
        self.N = 1000
         
        # Calculate/initialise some data-dependent parameters for training. We use
        # the intialisation of weights and biases as per Hinton (2010) practical
        # recommendations noting that the biases are initialised low for sparsity.
        # May change this....
        
        print("Model parameters initialising...")
        self.param_init()
        
        
        # GPU allocations
        # Set up some dedicated GPU memory for pseudorandom number generation
        print("Allocating GPU resources...")
        self.prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)             
        self.d_noise = cuda.device_array((self.R)) 
        
        self.d_gW = cuda.device_array((self.R,self.K))            # weight gradient
        self.d_gb = cuda.device_array((self.R))                   # visible bias gradient
        self.d_gc = cuda.device_array((self.K))                   # hidden bias gradient
        
        print("Initialisation complete")
        print("")
    
    
    
    def main(self): 
        # Training loop
        print("Begin training...")
        
        # We are going to time this for primitive profile-sake
        start = timer()
        
        q = np.zeros((self.K))                  # tracked estimate of the mean hidden activation probability
        
        for epoch in np.arange(self.max_epochs):
            
            self.alpha_t = self.alpha*(self.max_epochs-epoch)/self.max_epochs
            for self.B in np.arange(self.n_batches):
               
                self.d_gW = self.reset_gradients(self.d_gW)
                self.d_gb = self.reset_gradients(self.d_gb)
                self.d_gc = self.reset_gradients(self.d_gc)
                              
                for i in np.arange(self.B*self.batch_size,min((self.B+1)*self.batch_size,self.N)):
                              
                    # Calculate the expectation over the model distribtution using PCD
                    d_v = cuda.to_device(self.train[i,:])
                   
                    # Perform the Persistent CD algorithm
                    (Eh, h2, self.F) = self.PCD(self,d_v)
                    
                    # Update cumulative gradients and mean activation estimate
                    self.gW += np.outer(vi,Eh) - (np.einsum('ik,jk',self.F,h2)/self.PCD_size) # efficient way to evaluate sum of outer products
                    self.gb += vi - np.average(self.F,axis=1)
                    self.gc += Eh - np.average(h2,axis=1)
                    
                    q = self.l*q + (1-self.l)*Eh
            
                self.update_weights(q)
                
                self.W_size[epoch] = np.linalg.norm(self.W,'fro')
            if (epoch%self.t == 0):    
                print("Iteration: %d \t |W|_F: %.3f \t |b|_F: %.3f \t |c|_F: %.3f" \
                    % (epoch, self.W_size[epoch], np.linalg.norm(self.b), np.linalg.norm(self.c)))
        
        end = timer() - start
        print("Time = %g", end)
        
        # Save data to file
        self.save()
        
        # Visualise weights as a grid
        self.visualise()
        
    
    
    
    def load_data(self):
        '''
        Load the data --- I'm hoping to add extra options in future
        
        CREATES
        self.train      matrix of training data - each row is a training vector
        self.N          number of training samples
        self.R          number of dimensions of training 
        '''
        
        print("Loading data")
        f = np.load(self.inFile)
        self.train = f['digits']
        
        (self.N,self.R) = self.train.shape
        print("Number of training samples = %d" % self.N)
        print("Number of dimensions = %d" % self.R)
    
    
    
    def param_init(self):
        '''
        Initialise model parameters and storage --- will need to add GPU functionality later
        
        CREATES
        self.n_batches  number of mini-batches
        self.d_W        weight matrix
        self.d_b        visible biases
        self.d_c        hidden biases
        self.W_size     storage for Frobenius norm of weights
        self.d_F        fanstasy particle storage
        '''
        
        self.n_batches = np.ceil(self.N/self.batch_size)
        
        self.d_W = cuda.to_device(self.W)
        self.d_b = cuda.to_device(self.b)
        self.d_c = cuda.to_device(self.c)
        
        self.prng.normal(self.d_W, 0., 0.01)
        self.prng.normal(self.d_b, 0., 0.01)
        self.prng.normal(self.d_c, 0., 0.01)
        
        self.W_size = np.zeros((self.max_epochs,1))
        self.d_F = cuda.device_array((self.R,self.PCD_size))
        
    
    
    @vectorize(['float32(float32,float32)'], target='gpu')
    def sig(self, x):
        
        # Evaluation of the sigmoid nonlinearity on each element of the input list
        return 1./(1 + math.exp(-x))
    
    
    
    def bern_samp(self, m, h):
    
        # Draw a sample from the bernoulli distribution with mean m of length h. For
        # vector input each entry i can be interpreted as coming from an independent
        # bernoulli with mean m[i]. Note column vectors only.
        self.prng.uniform(self.d_noise)
        self.d_noise.copy_to_host(self.noise_mat)
        noise_mat2 = np.reshape(self.noise_mat,(self.R))
        return (noise_mat2 < m) * 1  
    
    
    def bern_samp_mat(self, m,(h,w)):
        
        # For a (h,w)-matrix sample each element iid from the bernoulli distribution
        # with matrix of means m[i,j].
        return (np.random.random_sample((h,w)) < m) * 1
    
    
    
    def sparsity(self, h):
        
        # h is the vector of hidden units
        # rho is the averrage sparsity penalty, say 0.05
        #
        # Calculate the derivative sparsity penality for the KL divergence sparsity.
        # I don't see how we could use L1 given that its derivative is signed ---
        # Perhaps I better revise this or check out the Francis Bach literature.
        #
        # Note, instead of/as well as using this penalty you can initialise the biases
        # at -4 to encourage bias sparsity.
        r = np.average(h)
        return -(self.rho/r) + ((1-self.rho)/(1-r))
    
    
    
    @jit(argtypes=[float32,float32], target='gpu')
    def PCD(self,v):
    
    # b is the column vector of visible biases
    # c is the column vector of hidden biases
    # W is the matrix of weights
    # F is the matrix of fantasy particles stored columnwise
    # v is the columm vector of the data
    # n is the number of fantasy particles in F
    # K is the number of hidden units
    # R is the number of visible units
        g = self.d_c + np.dot(v,self.d_W).T
        Eh = self.sig(g)
    
        ph = self.sig(self.c + np.dot(self.F.T,self.d_W)).T
        pv = self.sig(self.d_b + np.dot(self.W,ph).T).T            
        vsmpl = self.bern_samp_mat(pv,(self.R,self.PCD_size))       
        
        return Eh, ph, vsmpl
    
    
    
    def update_weights(self, q):
        # Update weights and biases, note the weight decay term
        self.batch_size2 = min((self.B+1)*self.batch_size,self.N) - self.B*self.batch_size
        
        self.W += self.alpha_t*(self.gW/self.batch_size2 - self.beta*self.sparsity(q))
        self.b += self.alpha_t*self.gb/self.batch_size2 
        self.c += self.alpha_t*(self.gc/self.batch_size2 - self.beta*self.sparsity(q))
    
    
    
    @vectorize(['float32(float32)'], target='gpu')
    def reset_gradients(vec):
        '''
        Reset all of the weight/bias gradients on the gpu
        '''
        return 0.
        
    
    
    def save(self):
        
        # Save parameters to file and create directory if it doesn't exist
        # Check if output file exists
        if self.outFile[-1] == '/':
            self.outFile = self.outFile[:-1]
        
        baseName = os.path.basename(self.outFile)
        dirName = os.path.dirname(self.outFile)
        
        # If not then create file
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        
        # Print to file
        np.savez(self.outFile,b=self.b,c=self.c,W=self.W)
        print("Data printed to file %r" % self.outFile)
        
        
        # Now to save the parameter statistics
        if self.outFile2[-1] == '/':
            self.outFile2 = self.outFile2[:-1]
        
        baseName = os.path.basename(self.outFile2)
        dirName = os.path.dirname(self.outFile2)
        
        # If not then create file
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
    
        np.savez(self.outFile2,W_size=self.W_size)
        
        print("Stats printed to file %r" % self.outFile2)
        
        
    def visualise(self):
        '''
        Visualise the weight matrices
        '''
        
        print("Loading weights for visualisation")
        sqrtK = np.ceil(np.sqrt(self.K))
        for k in np.arange(self.K):
            plt.subplot(sqrtK,sqrtK,k+1)
            img = self.W[:,k].reshape((np.sqrt(self.R),np.sqrt(self.R)))
            plt.imshow(img, cmap=plt.cm.gray,interpolation='nearest')
            plt.axis('off')
        # Show
        plt.show()

if __name__ == '__main__':
    rbm = Rbm(1)
    rbm.main()











































