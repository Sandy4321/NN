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
from PIL import Image
import pickle


# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')
dh.get_corrupt(corruption_level=0.2)
# Unpickle machine
print('Unpickling machine')
stream = open('AE.pkl','r')
AE = pickle.load(stream)
AE.data = dh

index = T.lscalar()  # index to a [mini]batch
AE_out = theano.function([index],
                    AE.output,
                    givens = {AE.x: AE.data.corrupt_set_x[index: (index + 100)]})

image = Image.fromarray(utils.tile_raster_images(X=AE_out(32),
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('AE_out.png')


#img = dh.test_set_x.get_value(borrow=True)[32:133,:]
#image = Image.fromarray(utils.tile_raster_images(X=img,
#             img_shape=(28,28), tile_shape=(10, 10),
#             tile_spacing=(1, 1)))
#image.save('num.png')


img = dh.corrupt_set_x.get_value(borrow=True)[32:133,:]
image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('num.png')

