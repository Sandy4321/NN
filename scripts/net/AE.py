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
topology = (784, 361, 196, 361, 784)
nonlinearities = ('sigmoid','sigmoid','sigmoid','sigmoid')
layer_types = ('DAE','DAE','DAE','DAE')
regularisation = (('xent','L2'),('xent','L2'),('xent','L2'),('xent','L2'))
device = 'AE'

# IO
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')
pkl_name = 'AE.pkl'
l0_filters = 'l0_filters.png'
l1_filters = 'l1_filters.png'

# Training parameters
initialisation_regime = 'Glorot'
optimisation_scheme='SDG'
layer_scheme='DAE'
pretrain_learning_rate = 0.1
fine_tune_learning_rate = 0.1
tau = 100    # later I want to figure out tau adaptively
momentum = 0.95
training_size = dh.train_set_x.get_value().shape[0]
batch_size = 50
n_train_batches = training_size/batch_size
n_valid_batches = dh.valid_set_x.get_value().shape[0]/batch_size
pretrain_epochs = 10
max_epochs = 200
patience_increase = 1.5
corruption_level = 0.5
np_rng = np.random.RandomState(123)
theano_rng = RandomStreams(np_rng.randint(2 ** 30))
pkl_rate = 50


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
AE.initialise_weights(initialisation_regime)

# Load the pretraining parameters
AE.load_pretrain_params('AE_xent',
                        optimisation_scheme,
                        n_train_batches,
                        batch_size=batch_size,
                        pretrain_learning_rate=pretrain_learning_rate,
                        pretrain_epochs=pretrain_epochs,
                        corruption_level=corruption_level)

AE.load_fine_tuning_params('L2',
                           'SGD',
                           fine_tune_learning_rate=fine_tune_learning_rate,
                           max_epochs=max_epochs,
                           patience_increase=patience_increase,
                           n_train_batches=n_train_batches,
                           n_valid_batches=n_valid_batches,
                           batch_size=batch_size,
                           momentum=momentum,
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

image2 = Image.fromarray(utils.tile_raster_images(X=AE.net[1].W.get_value(borrow=True).T,
             img_shape=(19, 19), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image2.save('l1_filters.png')
































