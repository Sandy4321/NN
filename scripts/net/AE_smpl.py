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

AE.sample_AE(32, 50, 50, 0.05)

'''

img = dh.test_set_x.get_value(borrow=True)[32:33,:]
image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(1, 1),
             tile_spacing=(1, 1)))
image.save('sample_seed.png')


image = Image.fromarray(utils.tile_raster_images(X=AE_out,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('sample_out.png')

'''