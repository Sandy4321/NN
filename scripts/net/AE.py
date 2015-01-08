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
layer_types = ('AE','AE','AE','AE')
regularisation = (('xent','L2'),('xent','L2'),('xent','L2'),('xent','L2'))
device = 'AE'

# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')

# Training parameters
initialisation_regime = 'Glorot'
optimisation_scheme='SDG'
learning_rate = 0.1
training_size = dh.train_set_x.get_value().shape[0]
batch_size = 50
n_train_batches = training_size/batch_size
pretrain_epochs = 10
corruption_level = 0.
np_rng = np.random.RandomState(123)
theano_rng = RandomStreams(np_rng.randint(2 ** 30))


### 2 LOAD PARAMETER VALUES ###



# Build deep network
AE = Deep(
    topology=topology,
    nonlinearities=nonlinearities,
    layer_types=layer_types,
    device=device,
    regularisation=regularisation,
    data = dh
)

# Initialise network weights
AE.initialise_weights(initialisation_regime)

# Load the pretraining parameters
AE.load_pretrain_params('AE_xent',
                     n_train_batches,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     pretrain_epochs=pretrain_epochs,
                     corruption_level=corruption_level)



### 3 TRAINING ###

AE.pretrain(optimisation_scheme=optimisation_scheme)

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
































