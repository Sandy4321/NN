# rbm_header.py
#
# Daniel E. Worrall --- 27 Oct 2014
#
# Script containing all of the custom functions for the RBM to work

# Imports
import numpy as np                         # for matrix operations
import os                                  # for file writing

def sig(x):
    
    # Evaluation of the sigmoid nonlinearity on each element of the input list
    return 1./(1 + np.exp(-x))

def bern_samp(m,h):
    
    # Draw a sample from the bernoulli distribution with mean m of length h. For
    # vector input each entry i can be interpreted as coming from an independent
    # bernoulli with mean m[i]. Note column vectors only.
    return (np.random.random_sample((h)) < m) * 1

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

    with file(outFile,'w') as f:
        print >> f, " ".join(str(n)[1:-1] for n in b)       # print b
        print >> f, " ".join(str(n)[1:-1] for n in c)       # print c
        for index in np.arange(R):
            print >> f, " ".join(str(n) for n in W[index,:])  # print W one row at a time
    
    f.close()
    print "Data printed to file %r" % outFile
    
    
    # Now to save the parameter statistics
    if outFile2[-1] == '/':
        outFile2 = outFile2[:-1]
    
    baseName = os.path.basename(outFile2)
    dirName = os.path.dirname(outFile2)
    
    # If not then create file
    if not os.path.isdir(dirName):
        os.makedirs(dirName)

    with file(outFile2,'w') as f:
        print >> f, " ".join(str(n)[1:-1] for n in W_size)  # print W_size
    
    f.close()
    print "Stats printed to file %r" % outFile2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    