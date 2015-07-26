'''Pruning'''

import os, time, sys

import cPickle
import numpy

from mlp import Mlp
from theano import config as Tconf
from train import DivergenceError, Train

def objective(lr):
    args = {
        'algorithm' : 'SGD',
        'RMScoeff' : None,
        'RMSreg' : None,
        'mode' : 'training',
        'learning_type' : 'classification',
        'num_classes' : 10,
        'train_cost_type' : 'nll',
        'valid_cost_type' : 'accuracy',
        'layer_sizes' : (784, 800, 800, 10),
        'nonlinearities' : ('ReLU', 'ReLU', 'SoftMax'),
        'data_address' : './data/mnist.pkl.gz',
        'binarize': False,
        'learning_rate' : lr,
        'lr_multipliers' : {'b' : 2.},
        'learning_rate_margin' : (0,200,300),
        'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
        'momentum' : 0.9,
        'momentum_ramp' : 0,
        'batch_size' : 128,
        'num_epochs' : 500,
        'prior' : 'Gaussian',
        'prior_variance' : None,
        'num_components' : 1,
        'num_samples' : 1,
        'norm' : 'L2',
        'max_row_norm' : numpy.sqrt(15.),
        'zero_mean' : False,
        'sparsity' : None, 
        'dropout_dict' : None,
        'cov' : False,
        'validation_freq' : 10,
        'save_freq' : 50,
        'save_name' : 'pkl/preprunedmlp.pkl'
        }

    dropout_dict = {}
    for i in numpy.arange(len(args['nonlinearities'])):
        name = 'layer' + str(i)
        size = (args['layer_sizes'][i],1)
        if i == 0:
            # Need to cast to floatX or the computation gets pushed to the CPU
            prior = 0.8*numpy.ones(size).astype(Tconf.floatX)
        else:
            prior = 0.5*numpy.ones(size).astype(Tconf.floatX)
        sub_dict = { name : {'seed' : 234,
                             'type' : 'unbiased',
                             'values' : prior}}
        dropout_dict.update(sub_dict)
    args['dropout_dict'] = dropout_dict
        
    tr = Train()
    tr.construct(Mlp, args)
    tr.build(args)
    tr.load_data(args)
    monitor = tr.train(args)


lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
for lr in lrs:
    print('Learning rate: %f' % lr)
    try:
        objective(lr)
    except DivergenceError:
        pass













    
    
    
