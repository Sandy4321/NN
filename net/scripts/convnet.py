'''
An deep autoencoder script for the deep-net framework

@author: dew
@date: 6 Jan 2013
'''

from layer import Layer
from data_handling import Data_handling
from deep import Deep
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import Image
import pickle



### 1 DEFINE PARAMETERS ###

# Network parameters
topology        = (784, 400, 10)
nonlinearities  = ('ReLU','softmax')
layer_types     = ('FF','FF')
regularisation  = (('None','L2'),('None','L2'))
device          = 'DAE'
layer_scheme    = 'DAE'

# IO
stream      = open('data.pkl','r')
dh          = pickle.load(stream)
stream.close()
pkl_name    = '../temp/convnet.pkl'


# Training parameters
#Shared
np_rng                          = np.random.RandomState(12)
theano_rng                      = RandomStreams(np_rng.randint(2 ** 30))
pkl_rate                        = 50
training_size                   = dh.train_set_x.get_value().shape[0]
batch_size                      = 100
n_train_batches                 = training_size/batch_size
n_valid_batches                 = dh.valid_set_x.get_value().shape[0]/batch_size
noise_type                      = 'salt_and_pepper'
corruption_level                = 0.4
corruption_scheme               = 'anneal'
corruption_tau                  = 50



# Training parameters
#Shared
np_rng                          = np.random.RandomState(12)
theano_rng                      = RandomStreams(np_rng.randint(2 ** 30))
pkl_rate                        = 50
training_size                   = dh.train_set_x.get_value().shape[0]
batch_size                      = 100
n_train_batches                 = training_size/batch_size
n_valid_batches                 = dh.valid_set_x.get_value().shape[0]/batch_size
noise_type                      = 'salt_and_pepper'
corruption_level                = 0.4
corruption_scheme               = 'anneal'
corruption_tau                  = 50

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
CNN = Deep(
    topology        = topology,
    nonlinearities  = nonlinearities,
    layer_types     = layer_types,
    device          = device,
    regularisation  = regularisation,
    data            = dh,
    pkl_name        = pkl_name
)



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
AE.unsupervised_fine_tuning()







































