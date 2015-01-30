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
topology = (784, 2000, 784)
nonlinearities = ('tanh','logistic')
layer_types = ('DAE','DAE')
regularisation = (('None','L2'),('None','L2'))
device = 'DAE'
layer_scheme='DAE'

# IO
#stream = open('ZCA_data.pkl','r')
#dh = pickle.load(stream)
#dh = Data_handling()
#dh.load_data('./data/mnist.pkl.gz')
stream = open('data.pkl','r')
dh = pickle.load(stream)
stream.close()
pkl_name = 'examples/AE1.pkl'


l0_filters = 'l0_filters.png'
l1_filters = 'l1_filters.png'

# Training parameters
#Shared
initialisation_regime = 'Glorot'
np_rng = np.random.RandomState(123)
theano_rng = RandomStreams(np_rng.randint(2 ** 30))
pkl_rate = 50
training_size = dh.train_set_x.get_value().shape[0]
batch_size = 100
n_train_batches = training_size/batch_size
n_valid_batches = dh.valid_set_x.get_value().shape[0]/batch_size

# Pretrain
pretrain_optimisation_scheme    ='SDG'
pretrain_loss_type              = 'AE_xent'
pretrain_learning_rate          = 0.015062389723830663
pretrain_epochs                 = 40
noise_type                      = 'salt_and_pepper'
corruption_level                = 0.5

#Fine tune
fine_tune_optimisation_scheme   ='SDG'
fine_tune_loss_type             = 'xent'
fine_tune_learning_rate         = 0.06603559201252239 
tau                             = 33   
momentum                        = 0.7893854999695049
regularisation_weight           = 9.4839854289419e-06
h_track                         = 0.879630813575219
sparsity_target                 = 0.16451965545002675
activation_weight               = 2.3836078880048033e-06
patience_increase               = 2.0
max_epochs                      = 200


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

# Load the pretraining parameters
AE.load_pretrain_params(pretrain_loss_type,
                        pretrain_optimisation_scheme,
                        layer_scheme,
                        n_train_batches,
                        batch_size=batch_size,
                        pretrain_learning_rate=pretrain_learning_rate,
                        pretrain_epochs=pretrain_epochs,
                        initialisation_regime=initialisation_regime,
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









### WRAP UP AND TEST ###

print('Pickling machine')
stream = open(pkl_name,'w')
AE.data = []    # don't want to resave data
pickle.dump(AE, stream)


































