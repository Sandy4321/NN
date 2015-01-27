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
import os

# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')
dh.get_corrupt(corruption_level=0.2)

dir = 'AE_hyp'
hyp = '/hyp'
i = 0
pkl = '.pkl'

file_name = dir + hyp + str(i) + pkl

while os.path.isfile(file_name):

    # Unpickle machine
    print('Unpickling machine: %i' % i)
    stream = open(file_name,'r')
    AE = pickle.load(stream)
    AE.data = dh
    
    # Cost on unseen data
    z = theano.function([],
        AE.output,
        givens = {AE.x: AE.data.test_set_x})
    x = AE.data.test_set_x.get_value()
    cost = - np.mean(np.sum(x * np.log(z()+1e-6) + (1 - x) * np.log(1 - z()+1e-6), axis=1))
    #cost = np.max(z()), np.min(z())
    
    print('Cost = %g, ' % cost)
    
    # Next file
    del AE
    i +=1
    file_name = dir + hyp + str(i) + pkl




































