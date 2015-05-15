'''Experiments as per Wan et Al'''

import os, sys, time

import numpy

from mlp import Mlp
from theano import config as Tconf
from train import Train


args = {
    'algorithm' : 'RMSNAG',
    'RMScoeff' : 0.9,
    'RMSreg' : 1e-3,
    'learning_type' : 'classification',
    'num_classes' : 10,
    'train_cost_type' : 'nll',
    'subnet_cost_type' : 'factored_bernoulli',
    'valid_cost_type' : 'accuracy',
    'layer_sizes' : (784, 800, 800, 10),
    'nonlinearities' : ('ReLU', 'ReLU', 'SoftMax'),
    'period' : None,
    'deadband' : None,
    'data_address' : './data/mnist.pkl.gz',
    'binarize': False,
    'learning_rate' : 1e-4,
    'dropout_lr' : 1e-8,
    'lr_bias_multiplier' : 2.,
    'learning_rate_margin' : (0,200,300),
    'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
    'momentum' : 0.9,
    'momentum_ramp' : 0,
    'batch_size' : 128,
    'num_epochs' : 500,
    'norm' : 'L2',
    'max_row_norm' : 3.87,
    'dropout_dict' : None,
    'logit_anneal' : None,
    'validation_freq' : 5,
    'save_freq' : 100,
    'save_name' : None
    }


dropout_dict = {}
for i in numpy.arange(len(args['nonlinearities'])):
    name = 'layer' + str(i)

    if i == 0:
        # Need to cast to floatX or the computation gets pushed to the CPU
        prior = 0.8*numpy.ones((784,1)).astype(Tconf.floatX)
    else:
        prior = 0.5*numpy.ones((args['layer_sizes'][i],1)).astype(Tconf.floatX)
    sub_dict = {name : {'seed' : 234,
                        'type' : 'unbiased',
                        'values' : prior}}
    dropout_dict.update(sub_dict)
args['dropout_dict'] = dropout_dict

for i in numpy.arange(5):
    tr = Train()
    args['save_name'] = 'pkl/vardrop' + str(i) + '.pkl'
    tr.build(Mlp, args)
    tr.load_data(args)
    monitor = tr.train(args)

'''
Dropout: drop<num>.pkl
Variational dropout: vardrop<num>.pkl
'''














