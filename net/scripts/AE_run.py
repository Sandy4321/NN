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
stream = open('data.pkl','r')
dh = pickle.load(stream)
stream.close()


# Unpickle machine
print('Unpickling machine')
stream = open('../temp/AE2.pkl','r')
AE = pickle.load(stream)
AE.data = dh
AE.data.get_corrupt('salt_and_pepper', 0.4)

#----------------------------------------------------------------------
# WEIGHTS
print('Visualising weights')
# Weights
image = Image.fromarray(utils.tile_raster_images(X=AE.net[0].W.get_value().T,
             img_shape=(28, 28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('../temp/l0_filters.png')

#----------------------------------------------------------------------
# Denoising passes
print('Bottom-up pass')
# Denoising
index = T.lscalar()  # index to a [mini]batch
AE_out = theano.function([index],
                    AE.output,
                    givens = {AE.x: AE.data.snp_set_x[index: (index + 100)]})

# Print output
image = Image.fromarray(utils.tile_raster_images(X=AE_out(32),
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('../temp/denoise.png')

# Print input
img = AE.data.snp_set_x.get_value()[32:133,:]
image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('../temp/original.png')



print('Plotting weights histogram')
#----------------------------------------------------------------------
# Weights histograms

bin_edges = np.linspace(-0.6, 1.0, 100)
for i in xrange(AE.num_layers/2):
    W = AE.net[i].W.get_value(borrow=True).flatten()
    plt.hist(W, bin_edges)

plt.show()

for i in xrange(AE.num_layers/2):
    B = AE.net[i].b.get_value(borrow=True).flatten()
    plt.hist(B, bin_edges)

plt.show()


# does the layer corrupt work?
ip = T.matrix('ip')
corr = AE.net[0].get_corrupt(ip,0.2)

fn = theano.function([index],
                    corr,
                    givens = {ip: AE.data.test_set_x[index:(index+100), :]})

img = fn(32)
image = Image.fromarray(utils.tile_raster_images(X=img,
             img_shape=(28,28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('../temp/get_corrt.png')

































