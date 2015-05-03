"""Do a hyperparameter search for the autoencoder"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"


import os, sys, time

import cPickle
import gzip
import numpy

from train_dev import DivergenceError, Train
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
        rj = RejectionSample()
        lower, upper = rj.uniform()
        
        self.save_name = './pkl_lecun/dropprior' + str(experiment) +'.pkl'
        self.save_best = './pkl_lecun/best_dropprior' + str(experiment) +'.pkl'
        self.best_serial = './pkl_lecun/best_serial' + str(experiment) +'.pkl'
        
        args = {
            'algorithm' : 'SGD',
            'learning_type' : 'classification',
            'num_classes' : 10,
            'train_cost_type' : 'nll',
            'valid_cost_type' : 'accuracy',
            'layer_sizes' : (784, 800, 800, 10),
            'nonlinearities' : ('ReLU', 'ReLU', 'SoftMax'),
            'data_address' : './data/mnist.pkl.gz',
            'learning_rate' : hp.loguniform('learning_rate', numpy.log(1e-5),
                                            numpy.log(1e0)),
            'learning_rate_margin' : 600,
            'momentum' : hp.uniform('momentum', 0.88, 0.999),
            'batch_size' : hp.quniform('batch_size', 20, 200, 10),
            'num_epochs' : 600,
            'norm' : 'L2',
            'max_row_norm' : hp.uniform('max_row_norm', 3, 4),
            'dropout_dict' : None,
            'lower' : lower,
            'upper' : upper,
            'validation_freq' : 5,
            'save_freq' : 50,
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
            lower = args['lower']
            upper = args['upper']
            
            for i in numpy.arange(len(args['nonlinearities'])):
                name = 'layer' + str(i)
                ls = args['layer_sizes'][i]
                if i == 0:
                    # Need to cast to floatX or the computation gets pushed to the CPU
                    v = 0.8*numpy.ones((784,1)).astype(Tconf.floatX)
                else:
                    v = numpy.linspace(lower,upper,ls)[:,numpy.newaxis].astype(Tconf.floatX)
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


class RejectionSample():
    def __init__(self):
        pass
    
    def uniform(self):
        '''Draw a uniform distribution uniformly'''
        # Sample mu in [0.0,0.5]
        mu = 0
        var = numpy.inf
        # Rejection sample variance
        '''
        while var > (mu**2)/3:
            mu = numpy.random.random_sample()/2
            std = numpy.random.random_sample()/numpy.sqrt(12.)
            var = std**2
        '''
        mu = numpy.random.random_sample()/2
        var = 0
        # With prob. 0.5 lift mu into [0.5,1.0]
        if numpy.random.random_sample() < 0.5:
            mu = 1. - mu
        a = mu - numpy.sqrt(3*var)
        b = mu + numpy.sqrt(3*var)
        return (a,b)

if __name__ == '__main__':
    hd = Hyp_dropprior()
    for i in numpy.arange(100):
        hd.main(i)
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
