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
topology = (784, 400, 196, 400, 784)
nonlinearities = ('split_continuous','split_continuous','linear','linear')
layer_types = ('DAE','DAE','DAE','DAE')
regularisation = (('None','L2'),('None','L2'),('None','L2'),('None','L2'))
device = 'DAE'
layer_scheme='DAE'

# IO
stream = open('ZCA_data.pkl','r')
dh = pickle.load(stream)
#dh = Data_handling()
#dh.load_data('./data/mnist.pkl.gz')
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
pretrain_learning_rate = 3.39008126363606e-06
pretrain_epochs = 11
noise_type = 'gaussian'
corruption_level = 2.890043484644745

#Fine tune
fine_tune_optimisation_scheme='SDG'
fine_tune_loss_type = 'L2'
fine_tune_learning_rate = 4.480271215145136e-05 # Need to implement the code which fits this well
tau = 33    # later I want to figure out tau adaptively
momentum = 0.8731264040909384
regularisation_weight = 0.001373725401855339
h_track = 0.8761648416240407
sparsity_target = 0.029707816888876776
activation_weight = 0.0002759764654105294
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









### WRAP UP AND TEST ###

print('Pickling machine')
stream = open('AE.pkl','w')
AE.data = []    # don't want to resave data
pickle.dump(AE, stream)

print('Generating images of weights')
image = Image.fromarray(utils.tile_raster_images(X=AE.net[0].W.get_value(borrow=True).T,
             img_shape=(28, 28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('l0_filters.png')

weight_size = np.floor(np.sqrt(topology[1])).astype(int)

image2 = Image.fromarray(utils.tile_raster_images(X=AE.net[1].W.get_value(borrow=True).T,
             img_shape=(weight_size, weight_size), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image2.save('l1_filters.png')
































