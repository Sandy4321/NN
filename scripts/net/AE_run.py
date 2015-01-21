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
import matplotlib.pyplot as plt


# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')
dh.get_corrupt(corruption_level=0.2)
# Unpickle machine
print('Unpickling machine')
stream = open('AE.pkl','r')
AE = pickle.load(stream)
AE.data = dh

print('Visualising weights')
# Weights
image = Image.fromarray(utils.tile_raster_images(X=AE.net[0].W.get_value().T,
             img_shape=(28, 28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('l0_filters.png')

'''
weight_size = np.floor(np.sqrt(AE.topology[1])).astype(int)
image2 = Image.fromarray(utils.tile_raster_images(X=AE.net[1].W.get_value().T,
             img_shape=(weight_size, weight_size), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image2.save('l1_filters.png')
'''

print('Bottom-up pass')
# Denoising
index = T.lscalar()  # index to a [mini]batch
AE_out = theano.function([index],
                    AE.output,
                    givens = {AE.x: AE.data.corrupt_set_x[index: (index + 100)]})

image = Image.fromarray(utils.tile_raster_images(X=AE_out(32),
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('denoise.png')

img = dh.corrupt_set_x.get_value()[32:133,:]
image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('original.png')

AE_out2 = theano.function([],
                    AE.output,
                    givens = {AE.x: AE.data.corrupt_set_x})




print('Plotting weights histogram')
#----------------------------------------------------------------------
# Plot a 1D density example

bin_edges = np.linspace(-0.6, 0.6, 100)
for i in xrange(AE.num_layers/2):
    W = AE.net[i].W.get_value(borrow=True).flatten()
    plt.hist(W, bin_edges)

plt.show()

for i in xrange(AE.num_layers/2):
    B = AE.net[i].b.get_value(borrow=True).flatten()
    plt.hist(B, bin_edges)

plt.show()












































