'''
Script to read trial db
'''
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import pickle



stream = open('AE_hyp.pkl','r')
trials = pickle.load(stream)

for entry in trials.results:
    print entry
