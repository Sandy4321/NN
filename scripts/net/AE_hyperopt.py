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



def objective(args):
    try:
        fine_tune_learning_rate = args
        
        ### 1 DEFINE PARAMETERS ###
        
        # Network parameters
        topology = (784, 400, 196, 400, 784)
        nonlinearities = ('split_continuous','split_continuous','linear','linear')
        layer_types = ('DAE','DAE','DAE','DAE')
        regularisation = (('None','L2'),('None','L2'),('None','L2'),('None','L2'))
        device = 'DAE'
        layer_scheme='DAE'
        
        # IO
        #stream = open('ZCA_data.pkl','r')
        #dh = pickle.load(stream)
        #dh = Data_handling()
        #dh.load_data('./data/mnist.pkl.gz')
        #dh.get_corrupt(corruption_level=0.2)
        stream = open('data.pkl','r')
        dh = pickle.load(stream)
        stream.close()
        pkl_name = 'AE.pkl'
        
        
        l0_filters = 'l0_filters.png'
        l1_filters = 'l1_filters.png'
        
        # Training parameters
        #Shared
        initialisation_regime = 'Glorot'
        np_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        pkl_rate = 50
        training_size = dh.train_set_x.get_value().shape[0]
        batch_size = 50
        n_train_batches = training_size/batch_size
        n_valid_batches = dh.valid_set_x.get_value().shape[0]/batch_size
        
        # Pretrain
        pretrain_optimisation_scheme='SDG'
        pretrain_loss_type = 'AE_SE'
        pretrain_learning_rate = 0.0003
        pretrain_epochs = 15
        noise_type = 'gaussian'
        corruption_level = 2
        
        #Fine tune
        fine_tune_optimisation_scheme='SDG'
        fine_tune_loss_type = 'L2'
        fine_tune_learning_rate = fine_tune_learning_rate
        tau = 100    # later I want to figure out tau adaptively
        momentum = 0.5
        regularisation_weight = 0.0001
        h_track=0.95
        sparsity_target = 0.05
        activation_weight = 0.01
        patience_increase = 2.0
        max_epochs = 1000
        
        
        ### 2 LOAD PARAMETER VALUES ###
        
        # Build deep network
        AE = Deep(
            topology=topology,
            nonlinearities=nonlinearities,
            layer_types=layer_types,
            device=device,
            regularisation=regularisation,
            data = dh,
            pkl_name = pkl_name
        )
        
        # Initialise network weights
        AE.init_weights(initialisation_regime)
        
        # Load the pretraining parameters
        AE.load_pretrain_params(pretrain_loss_type,
                                pretrain_optimisation_scheme,
                                layer_scheme,
                                n_train_batches,
                                batch_size=batch_size,
                                pretrain_learning_rate=pretrain_learning_rate,
                                pretrain_epochs=pretrain_epochs,
                                noise_type=noise_type,
                                corruption_level=corruption_level)
        
        AE.load_fine_tuning_params(fine_tune_loss_type,
                                   fine_tune_optimisation_scheme,
                                   fine_tune_learning_rate=fine_tune_learning_rate,
                                   max_epochs=max_epochs,
                                   patience_increase=patience_increase,
                                   n_train_batches=n_train_batches,
                                   n_valid_batches=n_valid_batches,
                                   batch_size=batch_size,
                                   momentum=momentum,
                                   regularisation_weight=regularisation_weight,
                                   h_track=h_track,
                                   sparsity_target=sparsity_target,
                                   activation_weight=activation_weight,
                                   tau=tau,
                                   pkl_rate=pkl_rate)
        
        
        ### 3 TRAINING ###
        AE.pretrain()
        AE.unsupervised_fine_tuning()
        
        
        ### 4 TEST ###
        index = T.scalar()
        score = theano.function([],
            AE.output,
            givens = {AE.x: AE.data.test_set_x})
        cost = 0.5*np.mean((score() - AE.data.test_set_x.get_value())**2)
        
        return {'loss': cost,
                'status': STATUS_OK,
                'time': time.time()}
    except Exception, e:
        return {'status': STATUS_FAIL,
                'loss': np.inf,
                'time': time.time(),
                'exception': str(e)}


if __name__ == '__main__':
    
    trials = Trials()
    space = (hp.loguniform('x',np.log(1e-8),np.log(1e-1)))
    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials)
    
    print best
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















