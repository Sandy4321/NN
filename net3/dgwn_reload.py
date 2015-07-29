'''DWGN train file'''

import os, time, sys

import cPickle
import matplotlib.pyplot as plt
import numpy

from DGWN import Dgwn
from train import DivergenceError, Train

fname = './pkl/DGWN784_800_800_DC.pkl'
stream = open(fname, 'r')
state = cPickle.load(stream)
stream.close()
args = state['args']
monitor = state['monitor']
best_model = monitor['best_model']

N = 25
num_samples = [1,2,5,10,25]
x = numpy.logspace(-4.5,-2.5,num=N)

kl = numpy.zeros((N,len(num_samples)))
snr = numpy.zeros((N,len(num_samples)))

for i in numpy.arange(N):
    print i
    # KL pruning
    tr = Train()
    tr.load_state(Dgwn, fname)
    tr.model.prune(1.-x[i], 'KL')
    args['mode'] = 'validation'
    tr.load_data(args)
    for k,ns in enumerate(num_samples):
        args['num_samples'] = ns
        tr.build_validate(args) 
        kl[i,k] = tr.validate(args)
    # SNR pruning
    tr = Train()
    tr.load_state(Dgwn, fname)
    tr.model.prune(1.-x[i], 'KL')
    args['mode'] = 'validation'
    tr.load_data(args)
    for k,ns in enumerate(num_samples):
        args['num_samples'] = ns
        tr.build_validate(args) 
        snr[i,k] = tr.validate(args)
    numpy.savez('pkl/pruned784_800_800_DC.pkl', kl=kl, snr=snr)

kl = kl.mean(axis=1)
snr = snr.mean(axis=1)
plt.figure()
plt.loglog(x,1.-kl,'r')
plt.loglog(x,1.-snr,'b')
plt.show()

'''
x = numpy.logspace(-5.,-3.,num=25)
plt.figure()
names = ['pkl/pruned784_800_800_G4.pkl.npz',]
for name in names:
    data = numpy.load(name)
    k = data['kl']
    s = data['snr']
    kmax = numpy.amax(k)
    smax = numpy.amax(s)
    krel = k/kmax
    srel = s/smax
    plt.loglog(x,1.-krel)
    plt.loglog(x,1.-srel,'--')
plt.xlabel('Proportion of unpruned weights')
plt.ylabel('Realtive performance decrease above baseline')
plt.show()
'''











    
    
    
