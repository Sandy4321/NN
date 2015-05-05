'''Visualise the weights of a particular model'''

import cPickle
import numpy
import pylab
import utils

from matplotlib import pyplot as plt
from PIL import Image

fname = './train_dev.pkl'
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
    pylab.show()


fig = plt.figure()
plt.plot(monitor['train_cost'])
plt.show()

for arg in args:
    if arg != 'dropout_dict':
        print arg, args[arg]

n = 30
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
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
