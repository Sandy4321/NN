'''Visualise the weights of a particular model'''

import cPickle
import numpy
import pylab
import utils

from matplotlib import pyplot as plt
from PIL import Image

for i in numpy.arange(1):
    fname = './pkl/DWGNreg.pkl'
    stream = open(fname, 'r')
    state = cPickle.load(stream)
    stream.close()
    
    monitor = state['monitor']
    args = state['hyperparams']
    print('Validation cost %f' % (monitor['best_cost'],))
    
    params = monitor['best_model']
    
    for param in params:
        print param
        print('Max: %f' % (numpy.amax(param.get_value()),))
        print('Min: %f' % (numpy.amin(param.get_value()),))
        pylab.figure()
        pylab.hist(param.get_value().flatten(), 50, normed=1)
        pylab.suptitle(param, fontsize=20)
        pylab.show()
    
    train_cost = monitor['train_cost']
    tc = numpy.zeros((0,1))
    for cost in train_cost:
        tc = numpy.vstack((tc,cost[:,numpy.newaxis]))
    fig = plt.figure()
    plt.plot(tc,'r')
    plt.ylim([2.,4.])
    plt.show()
    
    fig = plt.figure()
    plt.semilogx(monitor['valid_cost'],'b')
    plt.ylim([0.9,1.])
    plt.show()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
