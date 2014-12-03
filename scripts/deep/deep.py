'''
This builds on the now defunct RBM script. The basic areas can be split into
Topology, Training, Regularisation, IO, Sampling and Evaluations

@author: dew
@date: 27 Nov 14
'''

import numpy as np
import pycuda.gpuarray as ga
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.curandom import XORWOWRandomNumberGenerator 
import time

class Deep:
    
    def __init__(self):
        pass

    def __init__(self, top, trn, reg, io):
        
        # Load parameters
        self.top = top
        self.trn = trn
        self.reg = reg
        self.io = io
        print('Setting loaded')
    
    
        
    def load_data(self):
        '''
        Load the data --- I'm hoping to add extra options in future
        
        CREATES
        self.data       matrix of training data - each row is a training vector
        self.N          number of training samples
        '''
        
        print("Loading data")
        f = np.load(self.io.input)
        self.data = f['digits']
        (self.N , self.R) = self.data.shape
        
        print("Number of training samples = %d" % self.N)
        if self.R != self.top.layers[0]:
            print('Input layer has incorrect size')
            exit(1)
    
    
    
    def train(self):
        '''
        The training depends on the method selected. For now, we draw from the class
        of stochastic gradient descent methods SGDs. PCD and fPCD are the main algorithms.
        '''
        
        # We are going to place a time limit on the programme. When the elapsed time
        # exceeds a set threshold we let the alogrithm finish the current cycle and
        # then save the data
        start = time.time()
        
        # Clever GPU memory allocation can get us mega speed ups
        self.cr = XORWOWRandomNumberGenerator()
        self.GPU_allocate()
               
        # Cycle
        for epoch in np.arange(self.trn.max_epochs):
            # In here is a single training cycle
            Eh = self.PCD(epoch)
            print('Epoch = %d \t time elapsed = %0.2g' % (epoch,time.time()-start))
    
    
    
    def GPU_allocate(self):
        # Allocate GPU memory. The d_ stands for 'device side'
        print('Allocating GPU memory')
        K = self.top.layers[1]
        R = self.R
        N = self.N
        
        self.d_W = ga.empty((R,K), np.float32, order="C")      # weights
        self.d_b = ga.empty((R), np.float32, order="C")        # visible bias
        self.d_c = ga.empty((N), np.float32, order="C")        # hidden bias
        
        self.d_gW = ga.empty((R,K), np.float32, order="C")     # weight gradient
        self.d_gb = ga.empty((R), np.float32, order="C")       # visible bias gradient
        self.d_gc = ga.empty((N), np.float32, order="C")       # hidden bias gradient
        
        self.d_v = ga.empty((R), np.float32, order="C")        # visible vectors
        self.d_h = ga.empty((K), np.float32, order="C")        # hidden vectors
        self.d_data = ga.to_gpu(self.data.astype(np.float32, order="C"))          # REMEMBER TO CHANGE THIS LATER - works for small data sets
        
            
        # Now initialise array values randomly to instantiate symmetry breaking
        self.cr.fill_normal(self.d_W)
        self.cr.fill_normal(self.d_b)
        self.cr.fill_normal(self.d_c)
        
        self.cr.fill_normal(self.d_gW)
        self.cr.fill_normal(self.d_gb)
        self.cr.fill_normal(self.d_gc)
        
        self.d_W *= 0.01
        self.d_b *= 0.01
        self.d_c *= 0.01
        
        self.d_gW *= 0.01
        self.d_gb *= 0.01
        self.d_gc *= 0.01
        
        
    def PCD(self, i):
        '''
        Perform a single PCD update
        
        i is the index of the current training example
        d_v is the training example
        Eh is the hidden variable posterior
        d_W is the matrix of weights
        '''
        # b is the column vector of visible biases
        # c is the column vector of hidden biases
        # W is the matrix of weights
        # F is the matrix of fantasy particles stored columnwise
        # v is the columm vector of the data
        # n is the number of fantasy particles in F
        # K is the number of hidden units
        # R is the number of visible units
        i = int(i)
        self.d_v = self.d_data[i,:]
        self.d_h = ga.dot(self.d_W,self.d_v)
        
   
        print(self.d_W.shape)
        print(self.d_h)
        print(self.d_h.shape)
        
        #vv = self.d_v.get()
       # WW = self.W.get()
        
       # print 
    
        #ph = self.sig(self.c + np.dot(self.F.T,self.d_W)).T
        #pv = self.sig(self.d_b + np.dot(self.W,ph).T).T            
        #vsmpl = self.bern_samp_mat(pv,(self.R,self.PCD_size))       
        
       # return Eh
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    