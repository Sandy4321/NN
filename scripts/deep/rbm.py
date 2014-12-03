exit'''
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

TRN = namedtuple('TRN', ['alpha', 'num_particles', 'mini_batch_size', 'max_epochs', 'max_time'])
trn = TRN(np.exp(-1), 10.0, 10.0, 10, 3600)

REG = namedtuple('REG', ['weight_cost', 'sparsity_target', 'sparsity_decay'])
reg = REG(0.0005, 0.05, 0.95)

IO = namedtuple('IO', ['update_interval', 'save_interval', 'input', 'output', 'meta_out'])
io = IO(1, 5 , '../../data/binary_mnist.npz', '../../data/deep_out.npz', '../../data/deep_meta_out.npz')

# Now to instantiate a deep neural network object
deep = Deep(top, trn, reg, io)
deep.load_data()
deep.train()

    
