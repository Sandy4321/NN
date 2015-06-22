'''DWGN train file'''

import os, time, sys

import cPickle
import matplotlib.pyplot as plt
import numpy

from DGWN import Dgwn
from train import DivergenceError, Train

fname = './pkl/DGWN.pkl'
stream = open(fname, 'r')
state = cPickle.load(stream)
stream.close()
args = state['args']
monitor = state['monitor']
best_model = monitor['best_model']

N = 35
ns = 10
num_trials = 10
x = numpy.logspace(-5.,-3.,num=N)

kl = numpy.zeros((N,num_trials))
snr = numpy.zeros((N,num_trials))

for j in numpy.arange(num_trials):
    for i in numpy.arange(N):
        tr = Train()
        tr.load_state(Dgwn, fname)
        tr.model.prune(1.-x[i], 'KL')
        args['mode'] = 'validation'
        args['num_samples'] = ns
        tr.build_validate(args)
        tr.load_data(args)
        kl[i,j] = tr.validate(args)
        
        tr = Train()
        tr.load_state(Dgwn, fname)
        tr.model.prune(1.-x[i], 'SNR')
        args['mode'] = 'validation'
        args['num_samples'] = ns
        tr.build_validate(args)
        tr.load_data(args)
        snr[i,j] = tr.validate(args)
    numpy.savez('pkl/pruned2.pkl', kl=kl, snr=snr)

kl = kl.mean(axis=1)
snr = snr.mean(axis=1)
plt.figure()
plt.loglog(x,kl,'r')
plt.loglog(x,snr,'b')
plt.show()











    
    
    
