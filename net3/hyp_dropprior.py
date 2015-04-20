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


class Hyp_dropprior():
    def __init__(self):
        self.best_cost = numpy.inf
        self.save_name = 'dropprior.pkl'
        self.save_best = 'best_dropprior.pkl'
    
    def main(self):
        '''Sample hpyperparameter space and run'''
        trials = Trials()
        args = {
            'layer_sizes' : (784, 2000, 2000),
            'data_address' : './data/mnist.pkl.gz',
            'learning_rate' : hp.loguniform('learning_rate', numpy.log(1e-9),
                                            numpy.log(1e-1)),
            'momentum' : hp.uniform('momentum', 0, 0.999),
            'batch_size' : hp.quniform('batch_size', 20, 200, 20),
            'num_epochs' : 200,
            'validation_freq' : 25,
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
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    