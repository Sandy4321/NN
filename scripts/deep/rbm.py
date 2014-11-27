'''
A simple example of an RBM on the MNIST dataset

@author: dew
@date: 27 Nov 2014
'''

import numpy as np
from deep import Deep
from collections import namedtuple

# Initialise parameters - we use the named tuple to pass groups of parameters.

TOP = namedtuple('TOP', ['layers','types'])
top = TOP((784,250),('bin','bin'))
print('Topology set as %s' % top)

PCD = namedtuple('PCD', ['alpha', 'num_particles', 'mini_batch_size', 'max_epochs', 'max_time'])
pcd = PCD(np.exp(-1), 10.0, 10.0, 10, 3600)
print('Training parameters set')

REG = namedtuple('REG', ['weight_cost', 'sparsity_target', 'sparsity_decay'])
reg = REG(0.0005, 0.05, 0.95)
print('Regulariser set')

IO = namedtuple('IO', ['update_interval', 'save_interval'])
io = IO(1, 5)
print("IO settings set")

# Now to instantiate a deep neural network object
deep = Deep(top, pcd, reg, io)



#self.inFile = "../../data/preproc.npz"
#self.outFile = "../../data/params_mnist.npz"
#self.outFile2 = "../../data/params_stats_mnist.npz"


# Load data
#self.load_data()


    
