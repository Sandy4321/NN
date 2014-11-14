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

        # Localise later!!!!!!
        self.prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
        
        # Let's initialise pseudorandom noise generator on the the GPU
        # Setup pseudo-random noise generator --- will localise this later
            
        self.noise = np.zeros((784), dtype=np.float32)
        self.noise_mat = np.zeros((784,250), dtype=np.float32)
            
        self.d_noise = cuda.to_device(noise)
        self.d_noise_mat = cuda.to_device(noise_mat)
    
    def main(self):
    
        # Parameters
        self.alpha = np.exp(-1)             # learning rate numerator
        self.K = 250                 # num hidden units
        self.batch_size = 10.0       # mini-batch size
        self.max_epochs = 10        # number of epochs until finished
        self.t = 1                  # time between cmd line progress updates
        self.beta = 0.0005           # weight-cost (regularisation)
        self.rho = 0.05              # sparsity target
        self.l = 0.95                # sparsity decay
        self.PCD_size = 10           # number of fantasy particles
        self.inFile = "../../data/preproc.npz"
        self.outFile = "../../data/params_mnist.npz"
        self.outFile2 = "../../data/params_stats_mnist.npz"
        
        # Load data
        (self.train,(self.N,self.R)) = load_data(self.inFile)
        
        # Hack for analysis only
        self.train = self.train[:1000,:]
        self.N = 1000
        print("Data loaded")
        
           
        # Calculate/initialise some data-dependent parameters for training. We use
        # the intialisation of weights and biases as per Hinton (2010) practical
        # recommendations noting that the biases are initialised low for sparsity.
        # May change this....
        
        print("Parameters initialising...")
        (n_batches,W,b,c,W_size,q,F) = init(N,batch_size,R,K,max_epochs,PCD_size)
        print("Parameters initialised")
        
        # Training loop
        print("Begin training...")
        
        # We are going to time this for primitive profile-sake
        start = timer()
        
        for epoch in np.arange(max_epochs):
            alpha_t = alpha*(max_epochs-epoch)/max_epochs
            for B in np.arange(n_batches):
               
                gW = np.zeros((R,K))  # gradient of weights
                gb = np.zeros((R))    # visible bias gradient
                gc = np.zeros((K))    # hidden bias gradient
                
                for i in np.arange(B*batch_size,min((B+1)*batch_size,N)):
                              
                    # Calculate the expectation over the model distribtution using PCD
                    vi = train[i,:]
                   
                    # Perform the Persistent CD algorithm
                    (Eh, h2, F) = PCD(b,c,W,F,vi,PCD_size,K,R)
                    
                    # Update cumulative gradients and mean activation estimate
                    gW += np.outer(vi,Eh) - (np.einsum('ik,jk',F,h2)/PCD_size) # efficient way to evaluate sum of outer products
                    gb += vi - np.average(F,axis=1)
                    gc += Eh - np.average(h2,axis=1)
                    
                    q = l*q + (1-l)*Eh
            
                # Update weights and biases, note the weight decay term
                batch_size2 = min((B+1)*batch_size,N) - B*batch_size
                
                W += alpha_t*(gW/batch_size2 - beta*sparsity(q,rho))
                b += alpha_t*gb/batch_size2 
                c += alpha_t*(gc/batch_size2 - beta*sparsity(q,rho))
                
                W_size[epoch] = np.linalg.norm(W,'fro')
            if (epoch%t == 0):    
                print("Iteration: %d \t |W|_F: %.3f \t |b|_F: %.3f \t |c|_F: %.3f" \
                    % (epoch, W_size[epoch], np.linalg.norm(b), np.linalg.norm(c)))
        
        end = timer() - start
        print("Time = %g", end)
        
        # Save data to file
        save((W_size,b,c,W),outFile,outFile2)
        
        # Visualise weights as a grid
        visualise(K,R,W)
        
    
    
    
    def load_data(inFile):
        '''
        Load the data --- I'm hoping to add extra options in future
        '''
        
        print("Loading data")
        f = np.load(inFile)
        train = f['digits']
        
        (N,R) = train.shape
        print("Number of training samples =",N)
        print("Number of dimensions =",R)
        
        return (train,(N,R))
    
    
    
    def init(N,batch_size,R,K,max_epochs,PCD_size):
        '''
        Initialise data
        '''
        
        n_batches = np.ceil(N/batch_size)
        W = 0.01*np.random.randn(R,K)
        b = 0.01*np.random.randn(R)
        c = 0.01*np.random.randn(K)-4
        W_size = np.zeros((max_epochs,1))
        q = np.zeros((K))             # tracked estimate of the mean hidden activation probability
        F = 0.01*np.random.randn(R,PCD_size)
        
        return(n_batches,W,b,c,W_size,q,F)
    
    
    
    def visualise(K,R,W):
        '''
        Visualise the weight matrices
        '''
        
        print("Loading weights for visualisation")
        sqrtK = np.ceil(np.sqrt(K))
        for k in np.arange(K):
            plt.subplot(sqrtK,sqrtK,k)
            img = W[:,k].reshape((np.sqrt(R),np.sqrt(R)))
            plt.imshow(img, cmap=plt.cm.gray,interpolation='nearest')
            plt.axis('off')
        # Show
        plt.show()
    
    
    
    def sig(x):
        
        # Evaluation of the sigmoid nonlinearity on each element of the input list
        return 1./(1 + np.exp(-x))
    
    
    
    def bern_samp(m,h):
        
        # Draw a sample from the bernoulli distribution with mean m of length h. For
        # vector input each entry i can be interpreted as coming from an independent
        # bernoulli with mean m[i]. Note column vectors only.
        global prng
        global d_noise
        global noise
        prng.uniform(d_noise)
        d_noise.copy_to_host(noise)
        return (noise < m) * 1
    
    
    
    def bern_samp_mat(m,(h,w)):
        
        # For a (h,w)-matrix sample each element iid from the bernoulli distribution
        # with matrix of means m[i,j].
        return (np.random.random_sample((h,w)) < m) * 1
    
    
    
    def sparsity(h,rho):
        
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
        return -(rho/r) + ((1-rho)/(1-r))
        #return rho-r
    
    
    
    def CD(b,c,W,v,n,K):
        
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
            ph = sig(c + np.dot(vn,W))
            hn = bern_samp(ph,K)                  # sample of h (smpl induces bottleneck)
            vn = sig(b + np.dot(W,hn))            # probability of v (less noise than smpl)
        
        hn = sig(c + np.dot(vn,W))            # probability of h for final sweep
            
        return (Eh,vn,hn)
    
    
    
    def sample(b,c,W,v,n,K,R):
        
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
            ph = sig(c + np.dot(vn,W))
            hn = bern_samp(ph,K)                  # sample of h
            pv = sig(b + np.dot(W,hn))            
            vn = bern_samp(pv,R)                  # sample of v 
    
        #print sum(ph), sum(hn), sum(pv), sum(vn)
    
        return vn
    
    
    
    def PCD(b,c,W,F,v,n,K,R):
        
        # b is the column vector of visible biases
        # c is the column vector of hidden biases
        # W is the matrix of weights
        # F is the matrix of fantasy particles stored columnwise
        # v is the columm vector of the data
        # n is the number of fantasy particles in F
        # K is the number of hidden units
        # R is the number of visible units
        Eh = sig(c + np.dot(v,W).T)
    
        ph = sig(c + np.dot(F.T,W)).T
        pv = sig(b + np.dot(W,ph).T).T            
        vsmpl = bern_samp_mat(pv,(R,n))       
        
        return Eh, ph, vsmpl
    
    
    
    def save((W_size,b,c,W),outFile,outFile2):
        
        # Save parameters to file and create directory if it doesn't exist
        # Check if output file exists
        if outFile[-1] == '/':
            outFile = outFile[:-1]
        
        baseName = os.path.basename(outFile)
        dirName = os.path.dirname(outFile)
        
        # If not then create file
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        
        # Print to file
        (R,K) = W.shape
    
        np.savez(outFile,b=b,c=c,W=W)
        print("Data printed to file %r" % outFile)
        
        
        # Now to save the parameter statistics
        if outFile2[-1] == '/':
            outFile2 = outFile2[:-1]
        
        baseName = os.path.basename(outFile2)
        dirName = os.path.dirname(outFile2)
        
        # If not then create file
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
    
        np.savez(outFile2,W_size=W_size)
        
        print("Stats printed to file %r" % outFile2)
        
        

if __name__ == '__main__':
    rbm = Rbm(1)
    rbm.main()











































