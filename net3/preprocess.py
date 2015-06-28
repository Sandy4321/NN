'''Preprocess the input'''

import os,  sys, time

import numpy
import theano.tensor as T

class Preprocess():
    def __init__(self):
        pass
    
    def scale_unit8_float32(self, data):
        '''Scale data to the 0.-1. interval'''
        data = data/255.
        return data
    
    def whiten(self, data):
        '''Whiten the data using the ZCA transform'''
        pass
    
    def binarize(self, data):
        '''Binarize the data in [0.,1.] by threholding at 0.5'''
        data = data >= 0.5
        return data
    
    def zero_mean(self, data):
        '''Subtract individual image means'''
        data = data - numpy.mean(data,axis=0)
        return data