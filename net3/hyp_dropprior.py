"""Do a hyperparameter search for the autoencoder"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"


import os, sys, time

import cPickle
import gzip
import numpy

from dropprior import Autoencoder, DivergenceError, Train
from hyperopt import fmin, hp, STATUS_FAIL, STATUS_OK, tpe, Trials
from theano import config as Tconf


class Hyp_dropprior():
    def __init__(self):
        self.best_cost = numpy.inf
        self.save_name = './pkl/dropprior.pkl'
        self.save_best = './pkl/best_dropprior.pkl'
    
    def main(self):
        '''Sample hpyperparameter space and run'''
        trials = Trials()

        args = {
            'layer_sizes' : (784, 2000, 2000),
            'data_address' : './data/mnist.pkl.gz',
            'learning_rate' : hp.loguniform('learning_rate', numpy.log(1e-7),
                                            numpy.log(1e0)),
            'learning_rate_margin' : hp.uniform('learning_rate_margin', 20, 100),
            'momentum' : hp.uniform('momentum', 0.85, 0.9999),
            'batch_size' : hp.quniform('batch_size', 20, 200, 20),
            'num_epochs' : 200,
            'max_col_norm' : hp.uniform('max_col_norm', 2, 5),
            'validation_freq' : 5,
            'save_freq' : 50,
            'save_name' : self.save_name
            }

        best = fmin(self.objective,
                    space = args,
                    algo = tpe.suggest,
                    max_evals = 12,
                    trials = trials)
        
        print best
    
    def objective(self, args):
        '''The dropprior training procedure'''
        try:
            print args
            
            # Hyperopt does not allow us to pass nested dicts. We will have to
            # come up with a more elegant solution in future, but for now, we
            # simply define the dropout prior outside
            dropout_dict = {}
            for i in numpy.arange(4):
                name = 'layer' + str(i)
                if i == 0:
                    # Need to cast to floatX or the computation gets pushed to the CPU
                    v = 0.2*numpy.ones((784,1)).astype(Tconf.floatX)
                else:
                    v = 0.5*numpy.ones((2000,1)).astype(Tconf.floatX)
                sub_dict = { name : {'seed' : 234,
                                     'type' : 'unbiased',
                                     'values' : v}}
                dropout_dict.update(sub_dict)
            args['dropout_dict'] = dropout_dict
                
            tr = Train()
            model = Autoencoder
            tr.build(model, args)
            monitor = tr.train(args)
            print monitor['best_cost']
            return_dict =  {'loss' : monitor['best_cost'],
                            'status' : STATUS_OK }
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
    hd.main()
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    