'''Visualise the weights of a particular model'''

import cPickle
import numpy
import pylab
import utils

from matplotlib import pyplot as plt
from PIL import Image

for i in numpy.arange(1):
    fname = './pkl/DGWNrowGauss.pkl'
    stream = open(fname, 'r')
    state = cPickle.load(stream)
    stream.close()
    
    monitor = state['monitor']
    args = state['args']
    print('Validation cost %f' % (monitor['best_cost'],))
    
    params = monitor['best_model']
   
    for param in params:
        print param
        val = param.get_value()
        if 'R' in param.name:
            val = 1./(1. + numpy.exp(val))
        print('Max: %f' % (numpy.amax(val),))
        print('Min: %f' % (numpy.amin(val),))
        pylab.figure()
        pylab.hist(val.flatten(), 50, normed=1)
        pylab.suptitle(param, fontsize=20)
        pylab.show()
    
    train_cost = monitor['train_cost']
    tc = numpy.zeros((0,1))
    for cost in train_cost:
        tc = numpy.vstack((tc,cost[:,numpy.newaxis]))
    fig = plt.figure()
    plt.plot(tc,'r')
    #plt.ylim([2.,4.])
    plt.show()
    
    fig = plt.figure()
    plt.semilogx(monitor['valid_cost'],'b')
    plt.ylim([0.9,1.])
    plt.show()
    
    ''' 
    for i in numpy.arange(3):
        Mname = 'M' + str(i)
        j = [j for j, param in enumerate(params) if Mname == param.name][0]
        M_value = params[j].get_value()
        
        Rname = 'R' + str(i)
        j = [j for j, param in enumerate(params) if Rname == param.name][0]
        R_value = params[j].get_value()
        S_value = numpy.log(1. + numpy.exp(R_value))
        
        SNR = numpy.log(1e-6 + numpy.abs(M_value)/S_value)
        pylab.figure()
        pylab.hist(SNR.flatten(), 100, normed=1, histtype='step')
        pylab.show()
    ''' 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
