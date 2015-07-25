'''Pruning'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, time, sys

import cPickle
import numpy

from convnet import Convnet
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from theano import config as Tconf
from train import DivergenceError, Train

def objective(args):
    lr, mmtm = args
    for arg in args:
        print arg
    layers = {}
    layers['input'] = { 'name':'input', 'type':'data', 'shape':(128,1,28,28)}
    layers['fc1'] = { 'name':'fc1', 'input':'input', 'type':'fc', 'shape':(800,), 'nonlin':'ReLU', 'dropout' : 0.8, 'max_norm' : numpy.sqrt(15.)}
    layers['fc2'] = { 'name':'fc2', 'input':'fc1', 'type':'fc', 'shape':(800,), 'nonlin':'ReLU', 'dropout' : 0.5, 'max_norm' : numpy.sqrt(15.)}
    layers['output'] = { 'name':'output', 'input':'fc2', 'type':'fc', 'shape':(10,), 'nonlin':'SoftMax', 'dropout' : 0.5, 'max_norm' : numpy.sqrt(15.)}
    args = {
        'algorithm' : 'SGD',
        'RMScoeff' : None,
        'RMSreg' : None,
        'mode' : 'training',
        'learning_type' : 'classification',
        'num_classes' : 10,
        'train_cost_type' : 'nll',
        'valid_cost_type' : 'accuracy',
        'layers' : layers,
        'data_address' : './data/mnist.pkl.gz',
        'binarize': False,
        'zero_mean' : False,
        'learning_rate' : lr,
        'lr_multipliers' : {'b' : 2},
        'learning_rate_margin' : (0,200,300),
        'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
        'momentum' : mmtm,
        'momentum_ramp' : 0,
        'batch_size' : layers['input']['shape'][0],
        'num_epochs' : 500,
        'cov' : False,
        'validation_freq' : 2,
        'save_freq' : 10,
        'save_name' : 'pkl/preprunedmlp.pkl'
        }
     
    tr = Train()
    tr.construct(Convnet, args)
    tr.build(args)
    tr.load_data(args)
    monitor = tr.train(args)
    print(monitor['best_cost'])
        

if __name__ == '__main__':
    objective((1e-2, 0.9))










    
    
    
