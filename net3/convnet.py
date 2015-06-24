'''Multilayer perceptron model'''


__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import sys, time

import cPickle
import gzip
import numpy
import theano.tensor as T
import theano.tensor.nnet as Tnet

from matplotlib import pylab
from matplotlib import pyplot as plt
from theano import config as Tconf
from theano import function as Tfunction
from theano.tensor.nnet import conv as Tconv
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams


class Convnet():
    def __init__(self, args):
        '''Construct the MLP expression graph'''
        self.W = []
        self.b = []
        self.add_layers(args['layers'])
       
        
    def encode_layer(self, X, layer, args):
        '''Single layer'''
        nonlinearity = args['nonlinearities'][layer]
        name = 'layer' + str(layer)           
        # Dropout
        if self.dropout_dict == None:
            pre_act = T.dot(W, X) + self.b[layer]
        elif name in self.dropout_dict:
            G = self.dropout(layer, X.shape)
            self.G.append(G > 0)                    # To access mask values
            pre_act = T.dot(W, X*G) + self.b[layer]
        else:
            pre_act = T.dot(W, X) + self.b[layer]
        # Nonlinearity
        if nonlinearity == 'ReLU':
            s = lambda x : (x > 0) * x
        elif nonlinearity == 'SoftMax':
            s = Tnet.softmax
        else:
            print('Invalid nonlinearity')
            sys.exit(1) 
        return s(pre_act)
    
    def predict(self, X, args):
        '''Full MLP'''
        self.dropout_dict = args['dropout_dict']
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i, args)
            if args['cov'] == True:
                self.X[i] = X
                self.XXT[i] = T.dot(X,X.T)
        return (X,T.zeros_like(X[0,0]))
    
    def dropout(self, layer, size):
        '''Return a random dropout vector'''
        name = 'layer' + str(layer)
        if name in self.dropout_dict:
            sub_dict = self.dropout_dict[name]
            cseed = sub_dict['seed']
            # Construct RNG
            smrg = MRG_RandomStreams(seed=cseed)
            rng = smrg.uniform(size=size)
            # Evaluate RNG
            dropmult = (rng < self.q[layer]) / self.q[layer]
        return dropmult
    
    def add_layers(self, layers):
        '''Allocate memory for layers'''
        # Essentially the input is ALWAYS called input and output ALWAYS output
        # otherwise everything carries on from there like a linked list (LL). So
        # here we generate said LL.
        LL = self.linked_list(layers)
        # Now compute the filter sizes and add to storage
        input_shape = layers['input']['shape']
        for i in numpy.arange(len(LL)):
            filter_shape = LL[i]['shape']
            if 'stride' in LL[i]:
                stride = LL[i]['stride']
            if LL[i]['type'] == 'conv':
                self.add_conv(input_shape, filter_shape)
                output_shape = self.conv_size(input_shape, filter_shape, stride)
            elif LL[i]['type'] == 'pool':
                output_shape = self.pool_size(input_shape, filter_shape, stride)
            elif LL[i]['type'] == 'fc':
                self.add_fc(input_shape, filter_shape)
                output_shape = numpy.asarray(filter_shape)
            else:
                print('Invalid layer type')
                sys.exit(1)
            input_shape = output_shape
        for W in self.W:
            print W
            
    def linked_list(self, layers):
        '''Generate linked list'''
        output = layers['output']
        done = False
        LL = []
        LL.append(output)
        while(not done):
            LL.insert(0,layers[LL[0]['input']])
            if LL[0]['input'] == 'input':
                done = True
        return LL
    
    def conv_size(self, input_shape, filter_shape, stride=1):
        '''Compute the output shape for the conv layer'''
        input_hw = numpy.asarray(input_shape[2:])
        filter_hw = numpy.asarray(filter_shape[1:])
        output_hw = input_hw - filter_hw + numpy.ones((2,))
        output_shape = (input_shape[0], filter_shape[0], output_hw[0], output_hw[1])
        return numpy.asarray(output_shape)
    
    def pool_size(self, input_shape, pool, stride=None, pad=True):
        '''Compute the output shape'''
        output_shape = numpy.asarray(input_shape)
        if pad == True:
            output_shape[2] = numpy.ceil(float(output_shape[2])/pool[0])
            output_shape[3] = numpy.ceil(float(output_shape[3])/pool[1])
        else:
            output_shape[2] = numpy.floor(float(output_shape[2])/pool[0])
            output_shape[3] = numpy.floor(float(output_shape[3])/pool[1])
        return output_shape
    
    def add_conv(self, input_shape, filter_shape):
        '''Allocate memory for conv layers'''
        Wsh = (filter_shape[0], input_shape[1], filter_shape[1], filter_shape[2])
        coeff = 2.*numpy.sqrt(input_shape[1]*filter_shape[1]*filter_shape[2])
        W_value = coeff*(numpy.random.uniform(size=Wsh)-0.5)
        W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
        Wname = 'W' + str(len(self.W))
        self.W.append(TsharedX(W_value, Wname, borrow=True))   
        
        bsh = (filter_shape[0], )
        b_value = numpy.zeros(bsh)
        b_value = numpy.asarray(W_value, dtype=Tconf.floatX)
        bname = 'b' + str(len(self.b))
        self.b.append(TsharedX(b_value, bname, borrow=True))

    def add_fc(self, input_shape, filter_shape):
        '''Allocate memory for fc layer'''
        if input_shape.shape[0] > 1:
            Wsh = (input_shape[1:].prod(), filter_shape[0])
        else:
            Wsh = (input_shape, filter_shape[0])
        Wsh = numpy.asarray(Wsh, dtype=int)
        coeff = 2.*numpy.sqrt(numpy.asarray(Wsh).sum())
        W_value = coeff*(numpy.random.uniform(size=Wsh)-0.5)
        W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
        Wname = 'W' + str(len(self.W))
        self.W.append(TsharedX(W_value, Wname, borrow=True))
        
        bsh = (filter_shape[0], )
        b_value = 0.1*numpy.ones(bsh)
        b_value = numpy.asarray(W_value, dtype=Tconf.floatX)
        bname = 'b' + str(len(self.b))
        self.b.append(TsharedX(b_value, bname, borrow=True))
    
        
'''
TODO:
- DROPOUT

'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    