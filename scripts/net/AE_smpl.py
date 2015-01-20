'''
We see if we can construct a way of sampling from the AE

@author: dew
@date: 12 Jan 2013
'''

from layer import Layer
from data_handling import Data_handling
from deep import Deep
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import utils
from PIL import Image
import pickle


# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')
#dh.get_corrupt(corruption_level=0.1)
# Unpickle machine
print('Unpickling machine')
stream = open('AE.pkl','r')
AE = pickle.load(stream)
AE.data = dh

seed = np.asarray(dh.test_set_x.get_value()[383:384,:], dtype=theano.config.floatX)

burn_in = 500
num_samples = 500
corruption_level = 0.1
total_iter = burn_in+num_samples
vector_length = 28*28

AE_out = AE.sample_AE(seed, num_samples, burn_in, corruption_level)


img = seed
image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(1, 1),
             tile_spacing=(1, 1)))
image.save('sample_seed.png')

img = np.reshape(AE_out, (vector_length, total_iter+1), order='F')
grid_size = np.floor(np.sqrt(total_iter)).astype(int)

image = Image.fromarray(utils.tile_raster_images(X=img.T,
             img_shape=(28,28), tile_shape=(grid_size,grid_size),
             tile_spacing=(1, 1)))
image.save('sample_out.png')
