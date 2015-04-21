'''Visualise the weights of a particular model'''

import cPickle
import numpy
import pylab

from matplotlib import pyplot as plt

fname = './pkl/dropprior.pkl'
stream = open(fname, 'r')
state = cPickle.load(stream)
stream.close()

monitor = state['monitor']
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