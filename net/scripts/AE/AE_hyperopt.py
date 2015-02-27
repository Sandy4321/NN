'''
An deep autoencoder script for the deep-net framework

@author: dew
@date: 6 Jan 2013
'''

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from layer import Layer
from data_handling import Data_handling
from deep import Deep, DivergenceError
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import Image
import pickle
import time
import os


def objective(args):
    try:
        corruption_level, corruption_tau = args
        for i in args:
            print(i)
        
        ### 1 DEFINE PARAMETERS ###
        
        # Network parameters
        topology        = (784, 2000, 784)
        nonlinearities  = ('logistic','logistic')
        layer_types     = ('DAE','DAE')
        regularisation  = (('None','L2'),('None','L2'))
        device          = 'DAE'
        layer_scheme    = 'DAE'
        
        # IO
        stream      = open('data.pkl','r')
        dh          = pickle.load(stream)
        stream.close()
        pkl_name    = '../temp/AE1.pkl'
        
        
        l0_filters = 'l0_filters.png'
        l1_filters = 'l1_filters.png'
        
        # Training parameters
        #Shared
        initialisation_regime = 'Glorot'
        np_rng                          = np.random.RandomState(12)
        theano_rng                      = RandomStreams(np_rng.randint(2 ** 30))
        pkl_rate                        = 50
        training_size                   = dh.train_set_x.get_value().shape[0]
        batch_size                      = 100
        n_train_batches                 = training_size/batch_size
        n_valid_batches                 = dh.valid_set_x.get_value().shape[0]/batch_size
        noise_type                      = 'salt_and_pepper'
        corruption_level                = corruption_level
        corruption_scheme               = 'anneal'
        corruption_tau                  = np.int_(corruption_tau)
        
        # Pretrain
        pretrain_optimisation_scheme    = 'SDG'
        pretrain_loss_type              = 'L2'
        pretrain_learning_rate          = 10.0  
        pretrain_epochs                 = 1    
        
        #Fine tune
        fine_tune_optimisation_scheme   = 'SDG'
        fine_tune_loss_type             = 'L2'
        fine_tune_learning_rate         = 10.0    
        tau                             = 1     
        momentum                        = 0.   
        regularisation_weight           = 0.    
        h_track                         = 0.    
        sparsity_target                 = 0.   
        activation_weight               = 0.    
        patience_increase               = 2.0
        max_epochs                      = 200 - pretrain_epochs
        validation_frequency            = 5
        
        
        ### 2 LOAD PARAMETER VALUES ###
        
        # Build deep network
        AE = Deep(
            topology        = topology,
            nonlinearities  = nonlinearities,
            layer_types     = layer_types,
            device          = device,
            regularisation  = regularisation,
            data            = dh,
            pkl_name        = pkl_name
        )
        
        # Load the pretraining parameters
        AE.load_pretrain_params(pretrain_loss_type,
                                pretrain_optimisation_scheme,
                                layer_scheme,
                                n_train_batches,
                                batch_size                  = batch_size,
                                pretrain_learning_rate      = pretrain_learning_rate,
                                pretrain_epochs             = pretrain_epochs,
                                initialisation_regime       = initialisation_regime,
                                noise_type                  = noise_type,
                                corruption_level            = corruption_level,
                                corruption_scheme           = corruption_scheme,
                                corruption_tau              = corruption_tau)
        
        AE.load_fine_tuning_params(fine_tune_loss_type,
                                   fine_tune_optimisation_scheme,
                                   fine_tune_learning_rate  = fine_tune_learning_rate,
                                   max_epochs               = max_epochs,
                                   validation_frequency     = validation_frequency,
                                   patience_increase        = patience_increase,
                                   n_train_batches          = n_train_batches,
                                   n_valid_batches          = n_valid_batches,
                                   batch_size               = batch_size,
                                   momentum                 = momentum,
                                   regularisation_weight    = regularisation_weight,
                                   h_track                  = h_track,
                                   sparsity_target          = sparsity_target,
                                   activation_weight        = activation_weight,
                                   tau                      = tau,
                                   pkl_rate                 = pkl_rate,
                                   noise_type               = noise_type,
                                   corruption_level         = corruption_level,
                                   corruption_scheme        = corruption_scheme,
                                   corruption_tau           = corruption_tau)
        
        
        ### 3 TRAINING ###
        AE.pretrain()
        AE.unsupervised_fine_tuning()

        
        
        ### 4 TEST ###
        index   = T.scalar()
        score   = theano.function([],
            AE.output,
            givens = {AE.x: AE.data.corrupt_set_x})
        z       = score()
        x       = AE.data.test_set_x.get_value()
        cost	= np.mean(np.sum((x - z)**2, axis=1)) 
        
        ### 5 Store results ###
        fp  = '/home/daniel/Code/NN/net/AE_hyp'


        res = '/hyp'
        pkl = '.pkl'
        i   = 0
        file_name = fp + res + str(i) + pkl
        while os.path.isfile(file_name):
            i += 1
            file_name = fp + res + str(i) + pkl
        
        AE.pickle_machine(file_name)
        
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
    space = (hp.uniform('corruption_level',0.35,0.6),
             hp.quniform('corruption_tau', 1, 200, 1))
    best = fmin(objective,
        space       = space,
        algo        = tpe.suggest,
        max_evals   = 16, 
        trials      = trials)
    
    print best
    stream = open('AE_hyp.pkl','w')
    pickle.dump(trials, stream, pickle.HIGHEST_PROTOCOL)
    stream.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















