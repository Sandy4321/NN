'''Plot results of dropout mixture model'''
import os

import cPickle
import numpy 

from matplotlib import pyplot as plt


file_name = '/home/daniel/Code/NN/net3/pkl/DWGNreg.pkl'

stream = open(file_name, 'r')
state = cPickle.load(stream)
stream.close()

monitor = state['monitor']
args = state['hyperparams']
print('Validation cost %f' % (monitor['best_cost'],))

params = monitor['best_model']

M = {}
S = {}

for param in params:
    if 'M' in param.name:
        M[param.name] = param.get_value()
    elif 'R' in param.name:
        R = param.get_value()
        S[param.name] = numpy.log(1. + numpy.exp(R))

SNR = []

for i in numpy.arange(3):
    Mname = 'M' + str(i)
    Sname = 'S' + str(i)
    SNR.append(M[Mname]/S[Sname])
    
    fig = plt.figure()
    plt.hist(SNR[-1])
    plt.show()
























