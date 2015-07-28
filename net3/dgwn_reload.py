'''DWGN train file'''

import os, time, sys

import cPickle
import matplotlib.pyplot as plt
import numpy

from DGWN import Dgwn
from train import DivergenceError, Train

fname = './pkl/DGWN784_800_800_G3.pkl'
stream = open(fname, 'r')
state = cPickle.load(stream)
stream.close()
args = state['args']
monitor = state['monitor']
best_model = monitor['best_model']

N = 8
num_samples = [1,]
x = numpy.logspace(-1.,-0.,num=N)

kl = numpy.zeros((N,len(num_samples)))
snr = numpy.zeros((N,len(num_samples)))

for k,ns in enumerate(num_samples):
    for i in numpy.arange(N):
        print i
        tr = Train()
        tr.load_state(Dgwn, fname)
        tr.model.prune(1.-x[i], 'KL')
        args['mode'] = 'validation'
        args['num_samples'] = ns
        tr.build_validate(args)
        tr.load_data(args)
        kl[i,k] = tr.validate(args)
        
        tr = Train()
        tr.load_state(Dgwn, fname)
        tr.model.prune(1.-x[i], 'SNR')
        args['mode'] = 'validation'
        args['num_samples'] = ns
        tr.build_validate(args)
        tr.load_data(args)
        snr[i,k] = tr.validate(args)
    numpy.savez('pkl/pruned784_800_800_G3.pkl', kl=kl, snr=snr)

kl = kl.mean(axis=1)
snr = snr.mean(axis=1)
plt.figure()
plt.loglog(x,1.-kl,'r')
plt.loglog(x,1.-snr,'b')
plt.show()

'''
x = numpy.logspace(-4.,-0.,num=8)
plt.figure()
names = ['pkl/pruned784_800_800_G3.pkl.npz',]
for name in names:
    data = numpy.load(name)
    k = data['kl'].mean(axis=1)
    plt.loglog(x,k)
plt.xlabel('Proportion of unpruned weights')
plt.ylabel('Validation accurancy')
plt.show()
'''











    
    
    
