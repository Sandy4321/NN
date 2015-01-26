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
import sys



# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')
#dh.get_corrupt(corruption_level=0.1)
# Unpickle machine
print('Unpickling machine')
stream = open('AE.pkl','r')
AE = pickle.load(stream)
AE.data = dh

burn_in = 50
num_samples = 50
corruption_level = 0.08
total_iter = burn_in+num_samples
vector_length = 28*28
num_to_print = 10
stride = 1
batch_size = 10

start_point = 340
seed = np.asarray(dh.test_set_x.get_value()[start_point:start_point+batch_size,:], dtype=theano.config.floatX)

if num_to_print*stride > burn_in+num_samples+1:
    print('Sample range out of bounds')
    sys.exit(1)

print('Sampling')

AE_out = AE.sample_AE(seed, num_samples, burn_in, corruption_level)

print('Reshaping')

img = np.asarray([AE_out[:,:,i*stride] for i in xrange(num_to_print)])
print img.shape
img = np.reshape(img, (num_to_print*batch_size, vector_length), order='F')
print img.shape
grid_size = np.floor(np.sqrt(total_iter)).astype(int)

image = Image.fromarray(utils.tile_raster_images(X=img.T,
             img_shape=(28,28), tile_shape=(grid_size,grid_size),
             tile_spacing=(1, 1)))
image.save('sample_out.png')












































