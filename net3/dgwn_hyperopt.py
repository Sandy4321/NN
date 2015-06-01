'''DWGN train file'''

import os, time, sys

import cPickle
import numpy

from DGWN import Dgwn
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from train import DivergenceError, Train


def objective(oldargs):
    try:
        lr, mmtm, pv, ns = oldargs
        for arg in oldargs:
            print arg
        
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
            'lr_multipliers' : {'b' : 2., 'R' : 1e0},
            'learning_rate_margin' : (0,200,300),
            'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
            'momentum' : mmtm,
            'momentum_ramp' : 0,
            'batch_size' : 128,
            'num_epochs' : 500,
            'prior_variance' : pv,
            'num_components' : 1,
            'num_samples' : int(ns),
            'norm' : None,
            'max_row_norm' : None,
            'sparsity' : None, 
            'dropout_dict' : None,
            'cov' : False,
            'validation_freq' : 5,
            'save_freq' : 50,
            'save_name' : 'pkl/DGWNtest.pkl'
            }
        
        if args['sparsity'] != None:
            # Just for now until we sort ourselves out. Promise xx
            c = 1
            N = args['layer_sizes'][0]
            Y = args['layer_sizes'][-1]
            t = total_weights(args['layer_sizes'])
            H = layer_from_sparsity(N, Y, t, 1., 1-args['sparsity'], c)
            args['layer_sizes'] = write_neurons(N, H, Y, c)
            args['connectivity'] = (1.,) + (1-args['sparsity'],)*c + (1.,)
            args['nonlinearities'] = ('ReLU',) + ('ReLU',)*c + ('SoftMax',)
            print args['layer_sizes'], args['connectivity']
        
        if args['dropout_dict'] == True:
            dropout_dict = {}
            for i in numpy.arange(len(args['nonlinearities'])):
                name = 'layer' + str(i)
                shape = (args['layer_sizes'][i],1)
                if i == 0:
                    # Need to cast to floatX or the computation gets pushed to the CPU
                    prior = 0.8*numpy.ones(shape).astype(Tconf.floatX)
                else:
                    prior = 0.5*numpy.ones(shape).astype(Tconf.floatX)
                sub_dict = { name : {'seed' : 234,
                                     'values' : prior}}
                dropout_dict.update(sub_dict)
            args['dropout_dict'] = dropout_dict
            
        tr = Train()
        tr.build(Dgwn, args)
        tr.load_data(args)
        monitor = tr.train(args)
        cost = monitor['best_cost']
        
        i = 0
        file_name = './pkl/DWGNtest' + str(i) + '.pkl'
        while os.path.isfile(file_name):
            i += 1
            file_name = './pkl/DWGNtest' + str(i) + '.pkl'
        stream = open(file_name, 'w')
        cPickle.dump(monitor, stream, cPickle.HIGHEST_PROTOCOL)
        stream.close()
               
        return {'loss': cost,
                'status': STATUS_OK,
                'time': time.time()}
    except DivergenceError, e:
        return {'status': STATUS_FAIL,
                'loss': np.inf,
                'time': time.time(),
                'exception': str(e)}


if __name__ == '__main__':
    
    trials = Trials()

    space = (hp.loguniform('lr', numpy.log(1e-6), numpy.log(1e-2)),
             hp.uniform('hp', 0.85, 0.99),
             hp.loguniform('pv', numpy.log(1e-8), numpy.log(1e0)),
             hp.quniform('ns', 1, 10, 1))
    best = fmin(objective,
        space = space,
        algo = tpe.suggest,
        max_evals = 32, 
        trials = trials)
    
    print best
    stream = open('DGWN_hyp.pkl','w')
    cPickle.dump(trials, stream, cPickle.HIGHEST_PROTOCOL)
    stream.close()
    
    
    
    
    
    
