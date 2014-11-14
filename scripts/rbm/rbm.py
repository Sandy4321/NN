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

class Rbm:
    
    def __init__(self,parameters):
 
        # Initialise parameters. I want to include a nice wrapper later so that we can
        # nice and self-contained
        print("Training parameters loading")
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
    
        # Set up some dedicated GPU memory for pseudorandom number generation
       # self.prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
        
        # Let's initialise pseudorandom noise generator on the the GPU
        # Setup pseudo-random noise generator --- will localise this later
            
       # self.noise = np.zeros((784), dtype=np.float32)
       # self.noise_mat = np.zeros((784,250), dtype=np.float32)
            
      #  self.d_noise = cuda.to_device(noise)
       # self.d_noise_mat = cuda.to_device(noise_mat)
       
        # Load data
        self.load_data()
        
        # Hack for analysis only
        print("Data loading")
        self.train = self.train[:1000,:]
        self.N = 1000
         
        # Calculate/initialise some data-dependent parameters for training. We use
        # the intialisation of weights and biases as per Hinton (2010) practical
        # recommendations noting that the biases are initialised low for sparsity.
        # May change this....
        
        print("Model parameters initialising")
        self.param_init()

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
               
                gW = np.zeros((self.R,self.K))  # gradient of weights
                gb = np.zeros((self.R))         # visible bias gradient
                gc = np.zeros((self.K))    # hidden bias gradient
                
                for i in np.arange(self.B*self.batch_size,min((self.B+1)*self.batch_size,self.N)):
                              
                    # Calculate the expectation over the model distribtution using PCD
                    vi = self.train[i,:]
                   
                    # Perform the Persistent CD algorithm
                    (Eh, h2, self.F) = self.PCD(vi)
                    
                    # Update cumulative gradients and mean activation estimate
                    gW += np.outer(vi,Eh) - (np.einsum('ik,jk',self.F,h2)/self.PCD_size) # efficient way to evaluate sum of outer products
                    gb += vi - np.average(self.F,axis=1)
                    gc += Eh - np.average(h2,axis=1)
                    
                    q = self.l*q + (1-self.l)*Eh
            
                self.update_weights(gW,gb,gc,q)
                
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
        self.R          number of 
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
        self.W          weight matrix
        self.b          visible biases
        self.c          hidden biases
        self.W_size     storage for Frobenius norm of weights
        self.F          fanstasy particle storage
        '''
        
        self.n_batches = np.ceil(self.N/self.batch_size)
        self.W = 0.01*np.random.randn(self.R,self.K)
        self.b = 0.01*np.random.randn(self.R)
        self.c = 0.01*np.random.randn(self.K)-4
        self.W_size = np.zeros((self.max_epochs,1))
        self.F = 0.01*np.random.randn(self.R,self.PCD_size) 
    
    
    def sig(self, x):
        
        # Evaluation of the sigmoid nonlinearity on each element of the input list
        return 1./(1 + np.exp(-x))
    
    
    
    def bern_samp(self, m,h):
    
        # Draw a sample from the bernoulli distribution with mean m of length h. For
        # vector input each entry i can be interpreted as coming from an independent
        # bernoulli with mean m[i]. Note column vectors only.
        return (np.random.random_sample((h)) < m) * 1
    
    
    
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
    
    
    
    def CD(self, b,c,W,v,n,K):
        
        # b is the column vector of visible biases
        # c is the column vector of hidden biases
        # W is the matrix of weights
        # v is the columm vector of the data
        # n is the number of Gibbs iterations
        # K is the number of hidden units
        # 
        # Perform the n loads of constrastive divergence, known as CD-n
        # We should technically sample from the model distribution, but if we calculate
        # expectations instead then mixing is faster because we are rejecting the sample
        # noise. Reference Hinton 2010. 
        Eh = sig(c + np.dot(v,W))
        vn = v    
    
        for i in np.arange(n):
            ph = self.sig(c + np.dot(vn,W))
            hn = self.bern_samp(ph,K)                  # sample of h (smpl induces bottleneck)
            vn = self.sig(b + np.dot(W,hn))            # probability of v (less noise than smpl)
        
        hn = self.sig(c + np.dot(vn,W))            # probability of h for final sweep
            
        return (Eh,vn,hn)
    
    
    
    def sample(self, b,c,W,v,n,K,R):
        
        # b is the column vector of visible biases
        # c is the column vector of hidden biases
        # W is the matrix of weights
        # v is the columm vector of the seed data
        # n is the number of Gibbs iterations
        # K is the number of hidden units
        # R is the number of visible units
        # This is a really siple method which samples directly from the RBM given a set
        # of parameters and an initial state. The sampler will perform n iterations before
        # returning a sample --- increasing this value will create more decorrelated samples
        vn = v
        for i in np.arange(n):
            ph = self.sig(c + np.dot(vn,W))
            hn = self.bern_samp(ph,K)                  # sample of h
            pv = self.sig(b + np.dot(W,hn))            
            vn = self.bern_samp(pv,R)                  # sample of v 
    
        #print sum(ph), sum(hn), sum(pv), sum(vn)
    
        return vn
    
    
    
    def PCD(self,v):
        
        # b is the column vector of visible biases
        # c is the column vector of hidden biases
        # W is the matrix of weights
        # F is the matrix of fantasy particles stored columnwise
        # v is the columm vector of the data
        # n is the number of fantasy particles in F
        # K is the number of hidden units
        # R is the number of visible units
        Eh = self.sig(self.c + np.dot(v,self.W).T)
    
        ph = self.sig(self.c + np.dot(self.F.T,self.W)).T
        pv = self.sig(self.b + np.dot(self.W,ph).T).T            
        vsmpl = self.bern_samp_mat(pv,(self.R,self.PCD_size))       
        
        return Eh, ph, vsmpl
    
    
    
    def update_weights(self,gW,gb,gc,q):
        # Update weights and biases, note the weight decay term
        self.batch_size2 = min((self.B+1)*self.batch_size,self.N) - self.B*self.batch_size
        
        self.W += self.alpha_t*(gW/self.batch_size2 - self.beta*self.sparsity(q))
        self.b += self.alpha_t*gb/self.batch_size2 
        self.c += self.alpha_t*(gc/self.batch_size2 - self.beta*self.sparsity(q))
    
    
    
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











































