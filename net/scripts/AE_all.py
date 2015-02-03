'''
An deep autoencoder script for the deep-net framework. Run every hyp
file and find the best

@author: dew
@date: 6 Jan 2015
@updated 30 Jan 2015
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
import os

# Load dataset
stream = open('data.pkl','r')
dh = pickle.load(stream)
stream.close()

dir = '../AE_hyp'
hyp = '/hyp'
i = 0
pkl = '.pkl'

file_name = dir + hyp + str(i) + pkl
best = 0
best_cost = np.inf

while os.path.isfile(file_name):

    # Unpickle machine
    print('Machine: %i' % i),
    stream = open(file_name,'r')
    AE = pickle.load(stream)
    AE.data = dh
    
    # Cost on unseen data
    fn       = theano.function([],
        AE.output,
        givens = {AE.x: AE.data.corrupt_set_x})
    
    x       = AE.data.test_set_x.get_value()
    z       = fn()
    cost    = np.mean(0.5*np.sum((z - x)**2, axis=1))
    
    print('Cost = %g, ' % cost)
    
    if cost < best_cost:
        best = i
        best_cost = cost
        
    # PRINT DENOISE
    image = Image.fromarray(utils.tile_raster_images(X=z[32:132,:],
                 img_shape=(28,28), tile_shape=(10, 10),
                 tile_spacing=(1, 1)))
    image.save('../temp/denoise' + str(i) + '.png')
    
    #PRINT WEIGHTS
    image = Image.fromarray(utils.tile_raster_images(X=AE.net[0].W.get_value(borrow=True).T,
             img_shape=(28, 28), tile_shape=(20, 20),
             tile_spacing=(1, 1)))
    image.save('../temp/filters' + str(i) + '.png')
    
    # PRINT SAMPLES
    num_samples     = 300
    corruption_level= 0.4
    vector_length   = 28*28
    num_to_print    = 30
    stride          = 1
    batch_size      = 10
    noise_type      = 'salt_and_pepper'
    
    start_point     = 330
    seed            = np.asarray(dh.test_set_x.get_value()[start_point:start_point+batch_size,:] \
                                 , dtype=theano.config.floatX)
    
    if num_to_print*stride > num_samples+1:
        print('Sample range out of bounds')
        sys.exit(1)
    
    AE_out = AE.sample_AE(seed, num_samples, noise_type, corruption_level)
    
    img = np.zeros((batch_size*num_to_print,vector_length))
    for k in xrange(batch_size):
        for j in xrange(num_to_print):
            img[(k*num_to_print)+j,:] = AE_out[k,:,j*stride]
        
    image = Image.fromarray(utils.tile_raster_images(X=img,
                 img_shape=(28,28), tile_shape=(batch_size, num_to_print),
                 tile_spacing=(1, 1)))
    image.save('../temp/sample' + str(i) + '.png')
    
    
    # Next file
    i +=1
    del AE
    file_name = dir + hyp + str(i) + pkl

print('Best machine: %d' % best)




































