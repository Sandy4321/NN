'''DWGN train file'''

import os, time, sys

import cPickle
import numpy

from convnet import Convnet
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from theano import config as Tconf
from train import DivergenceError, Train

def objective(args):
    try:
        lr, mmtm, lrb = args
        for arg in args:
            print arg
        layers = {}
        layers['input'] = { 'name':'input', 'type':'data', 'shape':(100,1,28,28)}
        layers['conv0'] = { 'name':'conv0', 'input':'input', 'type':'conv', 'shape':(20,5,5), 'stride':(1,1), 'nonlin':'ReLU'}
        layers['pool1'] = { 'name':'pool1', 'input':'conv0', 'type':'pool', 'shape':(2,2)}
        #layers['conv2'] = { 'name':'conv2', 'input':'pool1', 'type':'conv', 'shape':(50,3,3), 'stride':(1,1), 'nonlin':'ReLU'}
        #layers['pool3'] = { 'name':'pool3', 'input':'conv2', 'type':'pool', 'shape':(2,2)}
        layers['fc4'] = { 'name':'fc4', 'input':'pool1', 'type':'fc', 'shape':(500,), 'nonlin':'ReLU', 'dropout' : 0.5, 'max_norm' : numpy.sqrt(15.)}
        layers['fc5'] = { 'name':'fc5', 'input':'fc4', 'type':'fc', 'shape':(500,), 'nonlin':'ReLU', 'dropout' : 0.5, 'max_norm' : numpy.sqrt(15.)}
        layers['output'] = { 'name':'output', 'input':'fc5', 'type':'fc', 'shape':(10,), 'nonlin':'SoftMax', 'dropout' : 0.5, 'max_norm' : numpy.sqrt(15.)}
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
            'lr_multipliers' : {'b' : lrb},
            'learning_rate_margin' : (0,700,900),
            'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
            'momentum' : mmtm,
            'momentum_ramp' : 0,
            'batch_size' : layers['input']['shape'][0],
            'num_epochs' : 1000,
            'cov' : False,
            'validation_freq' : 2,
            'save_freq' : 10,
            'save_name' : 'pkl/convnet.pkl'
            }
         
        tr = Train()
        tr.construct(Convnet, args)
        tr.build(args)
        tr.load_data(args)
        monitor = tr.train(args)
        cost = monitor['best_cost']
        
        return {'loss': -cost,
                'status': STATUS_OK}
    except DivergenceError, e:
        return {'status': STATUS_FAIL,
                'loss': numpy.inf,
                'exception': str(e)}
    

if __name__ == '__main__':
    '''
    trials = Trials()
    space = (hp.loguniform('lr', numpy.log(1e-9), numpy.log(1e-2)),
             hp.uniform('mmtm', 0.85, 0.99),
             hp.loguniform('lrb', numpy.log(1e-1), numpy.log(1e1)))
    best = fmin(objective,
                space = space,
                algo = tpe.suggest,
                max_evals = 16, 
                trials = trials)
    
    print best
    stream = open('bestconv.pkl','w')
    cPickle.dump(trials, stream, cPickle.HIGHEST_PROTOCOL)
    stream.close()
    '''
    objective((1e-2, 0.9, 2.))
# THERE IS SOMETHING WRONG WITH MY POOLING LAYERS!!!!










    
    
    
