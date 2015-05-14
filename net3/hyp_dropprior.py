"""Do a hyperparameter search for the autoencoder"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"


import os, sys, time

import cPickle
import gzip
import numpy

from train import DivergenceError, Train
from hyperopt import fmin, hp, STATUS_FAIL, STATUS_OK, tpe, Trials
from matplotlib import pyplot as plt
from mlp import Mlp
from theano import config as Tconf


class Hyp_dropprior():
    def __init__(self):
        self.best_cost = numpy.inf
    
    def main(self, experiment):
        '''Sample hyperparameter space and run'''
        trials = Trials()
        
        self.save_name = './pkl_lecun/dropprior' + str(experiment) +'.pkl'
        self.save_best = './pkl_lecun/best_dropprior' + str(experiment) +'.pkl'
        self.best_serial = './pkl_lecun/best_serial' + str(experiment) +'.pkl'
        
        args = {
                'algorithm' : 'SGD',
                'RMScoeff' : hp.uniform('RMScoeff', 0.85, 0.999),
                'RMSreg' : 1e-3,
                'learning_type' : 'classification',
                'num_classes' : 10,
                'train_cost_type' : 'nll',
                'valid_cost_type' : 'accuracy',
                'layer_sizes' : (784, 800, 800, 10),
                'nonlinearities' : ('wrapped_ReLU', 'wrapped_ReLU', 'SoftMax'),
                'period' : hp.uniform('period', 2., 10.),
                'deadband' : hp.uniform('deadband', 0., 1.),
                'data_address' : './data/mnist.pkl.gz',
                'learning_rate' : hp.loguniform('learning_rate', numpy.log(1e-4),
                                                    numpy.log(1e1)),
                'lr_bias_multiplier' : 2.,
                'learning_rate_margin' : (0,250,350),
                'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
                'momentum' : 0.9,
                'batch_size' : 128,
                'num_epochs' : 500,
                'norm' : 'L2',
                'max_row_norm' : hp.uniform('max_row_norm', 3, 4),
                'dropout_dict' : None, 
                'validation_freq' : 5,
                'save_freq' : 200,
                'save_name' : self.save_name
        }

        if args['valid_cost_type'] == 'accuracy':
            self.best_cost = -numpy.inf

        best = fmin(self.objective,
                    space = args,
                    algo = tpe.suggest,
                    max_evals = 16,
                    trials = trials)
        
        print best
        stream = open(self.best_serial, 'w')
        cPickle.dump(best, stream)
        stream.close()
    
    def objective(self, args):
        '''The dropprior training procedure'''
        try:
            for arg in args:
                print(arg, args[arg])
            
            # Hyperopt does not allow us to pass nested dicts. We will have to
            # come up with a more elegant solution in future, but for now, we
            # simply define the dropout prior outside
            dropout_dict = {}
            
            for i in numpy.arange(len(args['nonlinearities'])):
                name = 'layer' + str(i)
                ls = args['layer_sizes'][i]
                if i == 0:
                    # Need to cast to floatX or the computation gets pushed to the CPU
                    v = 1.*numpy.ones((784,1)).astype(Tconf.floatX)
                else:
                    v = 0.5*numpy.ones((args['layer_sizes'][i],1)).astype(Tconf.floatX)
                sub_dict = { name : {'seed' : 234,
                                     'type' : 'unbiased',
                                     'values' : v}}
                dropout_dict.update(sub_dict)
            args['dropout_dict'] = dropout_dict
            
            tr = Train()
            tr.build(Mlp, args)
            tr.load_data(args)
            monitor = tr.train(args)
            
            # If the validation channel is 'accuracy' we need to invert how we
            # judge a best model, because bigger is better.
            if args['valid_cost_type'] == 'accuracy':
                return_dict =  {'loss' : -monitor['best_cost'],
                                'status' : STATUS_OK }
                print -monitor['best_cost']
                if monitor['best_cost'] > self.best_cost:
                    stream = open(self.save_best, 'wb')
                    # We save the hyperparameters and the monitor
                    state = {'hyperparams' : args,
                             'monitor' : monitor}
                    cPickle.dump(state, stream, cPickle.HIGHEST_PROTOCOL)
                    stream.close()
                    self.best_cost = monitor['best_cost']
            elif args['valid_cost_type'] != 'accuracy':
                return_dict =  {'loss' : monitor['best_cost'],
                                'status' : STATUS_OK }
                print monitor['best_cost']
                if monitor['best_cost'] < self.best_cost:
                    stream = open(self.save_best, 'wb')
                    state = {'hyperparams' : args,
                             'monitor' : monitor}
                    cPickle.dump(state, stream, cPickle.HIGHEST_PROTOCOL)
                    stream.close()
                    self.best_cost = monitor['best_cost']
            
        except DivergenceError, e:
            return {'loss': numpy.inf,
                    'status': STATUS_FAIL,
                    'exception': str(e)}
        
        return return_dict


if __name__ == '__main__':
    hd = Hyp_dropprior()
    hd.main(1)
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
