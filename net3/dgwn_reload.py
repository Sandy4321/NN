'''DWGN train file'''

import os, time, sys

import cPickle
import numpy

from DGWN import Dgwn
from train import DivergenceError, Train

fname = './pkl/DGWNregm2.pkl'
stream = open(fname, 'r')
state = cPickle.load(stream)
stream.close()
args = state['args']
monitor = state['monitor']
best_model = monitor['best_model']


tr = Train()
tr.load_state(Dgwn, fname)
tr.build(args)
tr.load_data(args)
args['save_name'] = './pkl/DGWNreload.pkl'
monitor = tr.train(args)














    
    
    
