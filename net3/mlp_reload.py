'''DWGN train file'''

import os, time, sys

import cPickle
import matplotlib.pyplot as plt
import numpy

from mlp_dev import Mlp
from train import DivergenceError, Train

def get_reweights(tr, model, args):
    args['premean'] = True
    args['dropout_dict'] = None
    tr.build_validate(args)
    a = tr.validate(args, acts=True)
    return a


fname = './pkl/preprunedmlp.pkl'
stream = open(fname, 'r')
state = cPickle.load(stream)
stream.close()
args = state['args']
monitor = state['monitor']
best_model = monitor['best_model']

N = 25
ns = 1
num_trials = 1
x = numpy.logspace(-2.,-0.,num=N)
mode = 'info'

kl = numpy.zeros((N,num_trials))

for j in numpy.arange(num_trials):
    for i in numpy.arange(N):
        if mode == 'KL':
            tr = Train()
            tr.load_state(Mlp, fname)
            tr.load_data(args)
            tr.build_validate(args)
            tr.model.prune(1.-x[i], 'KL')
            args['mode'] = 'validation'
            args['num_samples'] = ns
            tr.build_validate(args)
            kl[i,j] = tr.validate(args)
        elif mode == 'info':
            tr = Train()
            tr.load_state(Mlp, './pkl/preprunedmlp.pkl')
            tr.load_data(args)
            r = get_reweights(tr, Mlp, args)
            tr.model.activation_pruning(1.-x[i], 'info', r)
            args['mode'] = 'validation'
            args['num_samples'] = ns
            tr.build_validate(args)
            kl[i,j] = tr.validate(args)
    numpy.savez('pkl/prunedmlpreweighted2.pkl', kl=kl)

kl = kl.mean(axis=1)
plt.figure()
plt.loglog(x,kl,'r')
plt.show()
'''
x = numpy.logspace(-2.,-0.,num=25)
plt.figure()
names = ['pkl/prunedmlp.pkl.npz','pkl/prunedmlpdrop1.pkl.npz', \
        'pkl/prunedmlpdrop2.pkl.npz','pkl/prunedmlpdrop5.pkl.npz', \
        'pkl/prunedmlpdrop10.pkl.npz','pkl/prunedmlpreweighted1.pkl.npz']
for name in names:
    data = numpy.load(name)
    k = data['kl'].mean(axis=1)
    kmax = k[-1]
    krel = k/kmax
    plt.loglog(x,1.-krel)
plt.xlabel('Proportion of unpruned weights')
plt.ylabel('Error above baseline')
plt.show()

'''






    
    
    
