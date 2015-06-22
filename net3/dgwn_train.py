'''DWGN train file'''

import os, time, sys

import cPickle
import numpy

from DGWN2 import Dgwn
from train import DivergenceError, Train

def objective(lr):
    args = {
        'algorithm' : 'SGD',
        'RMScoeff' : None,
        'RMSreg' : None,
        'mode' : 'training',
        'learning_type' : 'classification',
        'num_classes' : 10,
        'train_cost_type' : 'cross_entropy',
        'valid_cost_type' : 'accuracy',
        'layer_sizes' : (784, 800, 800, 10),
        'nonlinearities' : ('ReLU', 'ReLU', 'SoftMax'),
        'data_address' : './data/mnist.pkl.gz',
        'binarize': False,
        'learning_rate' : lr,
        'lr_multipliers' : {'b' : 2., 'R' : 1.},
        'learning_rate_margin' : (0,200,300),
        'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
        'momentum' : 0.9,
        'momentum_ramp' : 0,
        'batch_size' : 100,
        'num_epochs' : 500,
        'prior_variance' : 1e-3,
        'num_components' : 1,
        'num_samples' : 1,
        'norm' : None,
        'max_row_norm' : None,
        'sparsity' : None, 
        'dropout_dict' : None,
        'cov' : False,
        'validation_freq' : 10,
        'save_freq' : 50,
        'save_name' : 'pkl/DGWNBlundell.pkl'
        }
    
    tr = Train()
    tr.construct(Dgwn, args)
    tr.build(args)
    tr.load_data(args)
    monitor = tr.train(args)


lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

for lr in lrs:
    print('Learning rate: %f' % lr)
    try:
        objective(lr*0.01)
    except DivergenceError:
        pass













    
    
    
