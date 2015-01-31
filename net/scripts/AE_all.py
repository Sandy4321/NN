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
    z = theano.function([],
        AE.output,
        givens = {AE.x: AE.data.snp_set_x})
    x = AE.data.test_set_x.get_value()
    #cost = - np.mean(np.sum(x * np.log(z()+1e-9) + (1 - x) * np.log(1 - z()+1e-9), axis=1))
    cost = np.mean(0.5*np.sum((z() - x)**2, axis=1))
    
    print('Cost = %g, ' % cost)
    
    if cost < best_cost:
        best = i
        best_cost = cost
    
    # Next file
    i +=1
    del AE
    file_name = dir + hyp + str(i) + pkl

print('Best machine: %d' % best)




































