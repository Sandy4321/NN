'''Visualise the weights of a particular model'''

import cPickle
import numpy
import pylab
import utils

from matplotlib import pyplot as plt
from PIL import Image

for i in numpy.arange(1):
    fname = './pkl/DWGN.pkl'
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
    plt.show()
    
    """
    train_cost = monitor['max_grad']
    tc = numpy.zeros((0,1))
    for cost in train_cost:
        tc = numpy.vstack((tc,cost[:,numpy.newaxis]))
    fig = plt.figure()
    plt.plot(tc,'r')
    plt.show()
    
    max_q = monitor['max_q']
    min_q = monitor['min_q']
    mean_q = monitor['mean_q']
    iq = numpy.zeros((0,1))
    aq = numpy.zeros((0,1))
    eq = numpy.zeros((0,1))
    for i, a, e in zip(min_q, max_q, mean_q):
        iq = numpy.vstack((iq,i[:,numpy.newaxis]))
        aq = numpy.vstack((aq,a[:,numpy.newaxis]))
        eq = numpy.vstack((eq,e[:,numpy.newaxis]))
    fig = plt.figure()
    plt.plot(aq,'b')
    plt.plot(iq,'r')
    plt.plot(eq,'g')
    plt.show()
    
    train_cost = monitor['grad2']
    tc = numpy.zeros((0,1))
    for cost in train_cost:
        tc = numpy.vstack((tc,cost[:,numpy.newaxis]))
    fig = plt.figure()
    plt.plot(tc,'r')
    plt.show()
    """
    
    fig = plt.figure()
    plt.semilogx(monitor['valid_cost'],'b')
    plt.show()
    
    for param in hypparams:
        print param
        print('Max: %f' % (numpy.amax(param.get_value()),))
        print('Min: %f' % (numpy.amin(param.get_value()),))
        pylab.figure()
        n, bins, patches = pylab.hist(param.get_value().flatten(), 100, histtype='step')
        pylab.suptitle(param, fontsize=20)
        pylab.show()
    
    for arg in args:
        if arg != 'dropout_dict':
            print arg, args[arg]
    
    """
    n = 20
    for param in params:
        W = param.get_value()
        Wsh = W.shape
        if Wsh[1] > 1:
            idx = numpy.random.choice(Wsh[0], n**2, replace=False)
            print('Max: %f' % (numpy.amax(W),))
            print('Min: %f' % (numpy.amin(W),))
            w = W[idx,:]
            im = Image.fromarray(utils.tile_raster_images(X=w,
                                                          img_shape=(28,28),
                                                          tile_shape=(n,n),
                                                          tile_spacing=(1,1)))
            im.show()
    """
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
