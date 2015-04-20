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
dh.load_data('../data/mnist.pkl.gz')
# Unpickle machine
print('Unpickling machine')
stream = open('../AE_hyp/hyp10.pkl','r')
AE = pickle.load(stream)
AE.data = dh

num_samples     = 300
corruption_level= 0.3
vector_length   = 28*28
num_to_print    = 30
stride          = 1
batch_size      = 30
noise_type      = 'salt_and_pepper'

start_point     = 430
seed            = np.asarray(dh.test_set_x.get_value()[start_point:start_point+batch_size,:], dtype=theano.config.floatX)

if num_to_print*stride > num_samples+1:
    print('Sample range out of bounds')
    sys.exit(1)

print('Sampling')

AE_out = AE.sample_AE(seed, num_samples, noise_type, corruption_level)

print('Reshaping')

img = np.zeros((batch_size*num_to_print,vector_length))
for i in xrange(batch_size):
    for j in xrange(num_to_print):
        img[(i*num_to_print)+j,:] = AE_out[i,:,j*stride]



image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(batch_size, num_to_print),
             tile_spacing=(1, 1)))
image.save('../temp/sample_out.png')











































