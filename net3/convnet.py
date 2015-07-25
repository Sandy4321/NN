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
#from theano.sandbox.cuda import dnn as Tdnn
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample as Tdownsample


class Convnet():
    def __init__(self, args):
        '''Construct the MLP expression graph'''
        self.W = []
        self.b = []
        self._params = []
        self.add_layers(args['layers'])
        self.activations = {}
       
    def encode_layer(self, X, i, args):
        '''Single layer'''
        layer = self.LL[i]
        name = layer['name']
        if layer['type'] == 'conv':
            idx = self.get_idx(name)
            if 'dropout' in layer:
                G = self.dropout(layer, X.shape, layer['dropout'])
                X = X*(G>0)
            #pre_act = Tdnn.dnn_conv(X, self.W[idx], subsample=layer['stride']) + self.b[idx].dimshuffle('x',0,'x','x')
            pre_act = Tconv.conv2d(input=X, filters=self.W[idx]) + self.b[idx].dimshuffle('x',0,'x','x')
        elif layer['type'] == 'pool':
            if 'stride' in layer:
                pre_act = Tdownsample.max_pool_2d(X, ds=layer['shape'], st=layer['stride'], mode='max')
            else:
                pre_act = Tdownsample.max_pool_2d(X, ds=layer['shape'], mode='max')
            #pre_act = Tdnn.dnn_pool(X, layer['shape'], layer['stride'])
        elif layer['type'] == 'fc':
            idx = self.get_idx(name)
            #X = self.x2fc(X)
            if X.ndim > 2:
                X = X.flatten(2).T
            else:
                X = X.flatten(2)
            if 'dropout' in layer:
                G = self.dropout(layer, X.shape, layer['dropout'])
                X = X*(G>0)
            pre_act = T.dot(self.W[idx], X) + self.b[idx].dimshuffle(0,'x') 
        else:
            print('Invalid layer type')
            sys.exit(1)
        
        # Nonlinearity
        if 'nonlin' in layer:
            if layer['nonlin'] == 'ReLU':
                s = lambda x : (x > 0) * x
            elif layer['nonlin'] == 'SoftMax':
                s = Tnet.softmax
            else:
                print('Invalid nonlinearity')
                sys.exit(1)
        else:
            s = lambda x : x
        self.activations[name] = s(pre_act)
        return self.activations[name]
    
    def predict(self, X, args):
        '''Full MLP'''
        X = X.reshape(args['layers']['input']['shape'])
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i, args)
        return (X,T.zeros_like(X[0,0]))
    
    def dropout(self, layer, size, p):
        '''Return a random dropout vector'''
        assert p <= 1.
        assert p >= 0.
        # Construct RNG
        smrg = MRG_RandomStreams()
        rng = smrg.uniform(size=size)
        return (rng < p) / p
    
    def add_layers(self, layers):
        '''Allocate memory for layers'''
        # Essentially the input is ALWAYS called input and output ALWAYS output
        # otherwise everything carries on from there like a linked list (LL). So
        # here we generate said LL.
        LL = self.linked_list(layers)
        self.num_layers = len(LL)
        self.LL = LL
        # Now compute the filter sizes and add to storage
        input_shape = numpy.asarray(layers['input']['shape'])
        total_param = 0
        for i in numpy.arange(len(LL)):
            name = LL[i]['name']
            num_param = 0
            filter_shape = numpy.asarray(LL[i]['shape'])
            if 'stride' in LL[i]:
                stride = LL[i]['stride']
            else:
                stride = None
            if LL[i]['type'] == 'conv':
                num_param = self.add_conv(input_shape, filter_shape, name)
                output_shape = self.conv_size(input_shape, filter_shape, stride)
                print('%s \t In: %s \t Out: %s \t Filter:%s \t Num param: %i' %
                      (LL[i]['name'],input_shape,output_shape,filter_shape,num_param))
            elif LL[i]['type'] == 'pool':
                output_shape = self.pool_size(input_shape, filter_shape, stride)
                print('%s \t In: %s \t Out: %s \t Filter:%s' %
                      (LL[i]['name'],input_shape,output_shape,filter_shape))
            elif LL[i]['type'] == 'fc':
                if input_shape.shape[0] > 2:
                    filter_shape = numpy.asarray((filter_shape[0], input_shape[1:].prod()))
                    output_shape = numpy.asarray((filter_shape[0],input_shape[0]))
                else:
                    filter_shape = numpy.asarray((filter_shape[0], input_shape[0]))
                    output_shape = numpy.asarray((filter_shape[0],input_shape[1]))
                num_param = self.add_fc(input_shape, filter_shape, name)
                print('%s \t In: %s \t Out: %s \t Filter:%s \t Num param: %i' %
                      (LL[i]['name'],input_shape,output_shape,filter_shape,num_param))
            else:
                print('Invalid layer type')
                sys.exit(1)
            total_param += num_param
            input_shape = output_shape
            
        for W, b in zip(self.W, self.b):
            self._params.append(W)
            self._params.append(b)
        print('NUM PARAM: %i' % (total_param,))
            
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
    
    def conv_size(self, input_shape, filter_shape, stride=(1,1)):
        '''Compute the output shape for the conv layer'''
        input_hw = numpy.asarray(input_shape[2:])
        filter_hw = numpy.asarray(filter_shape[1:])
        output_hw = input_hw - filter_hw + numpy.ones((2,))
        output_shape = (input_shape[0], filter_shape[0], output_hw[0], output_hw[1])
        return numpy.asarray(output_shape, dtype=int)
    
    def pool_size(self, input_shape, pool, stride=None, pad=False):
        '''Compute the output shape'''
        output_shape = numpy.asarray(input_shape.copy(), dtype=int)
        if stride == None:
            if pad == True:
                output_shape[2] = numpy.ceil(float(output_shape[2])/pool[0])
                output_shape[3] = numpy.ceil(float(output_shape[3])/pool[1])
            else:
                output_shape[2] = numpy.floor(float(output_shape[2])/pool[0])
                output_shape[3] = numpy.floor(float(output_shape[3])/pool[1])
        else:
            output_shape[2] = output_shape[2] - pool[0] + stride[0] ### STRIDE=1 CHANGE
            output_shape[3] = output_shape[3] - pool[1] + stride[1]
        return output_shape
    
    def add_conv(self, input_shape, filter_shape, name):
        '''Allocate memory for conv layers'''
        Wsh = (filter_shape[0], input_shape[1], filter_shape[1], filter_shape[2])
        Wsh = numpy.asarray(Wsh, dtype=int)
        coeff = numpy.sqrt(256./(filter_shape.prod()))
        #coeff = 0.01
        W_value = coeff*numpy.random.normal(0., coeff, size=Wsh)
        W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
        Wname = 'W' + name
        self.W.append(TsharedX(W_value, Wname, borrow=True))
        bsh = (filter_shape[0], )
        b_value = 0.*numpy.ones(bsh)
        b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
        bname = 'b' + name
        self.b.append(TsharedX(b_value, bname, borrow=True))
        return Wsh.prod() + bsh[0]

    def add_fc(self, input_shape, filter_shape, name):
        '''Allocate memory for fc layer'''
        coeff = numpy.sqrt(2./(filter_shape.sum()))
        W_value = numpy.random.normal(0.,coeff, size=filter_shape)
        W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
        Wname = 'W' + name
        print numpy.var(W_value)
        self.W.append(TsharedX(W_value, Wname, borrow=True))
        bsh = (filter_shape[0], )
        b_value = 0.3*numpy.ones(bsh)
        b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
        bname = 'b' + name
        self.b.append(TsharedX(b_value, bname, borrow=True))
        return filter_shape.prod() + bsh[0]
    
    def get_idx(self, name):
        '''Get the layer index for the parameter list'''
        name = 'W' + name
        for i, W in enumerate(self.W):
            if W.name == name:
                idx = i
        return idx
    
    def x2fc(self, X):
        '''Convert X to fully connected form'''
        if X.ndim == 4:
            X = X.reshape((X.shape[1:].prod(),X.shape[0]))
        return X
    
'''
TODO:
- READOUTS
- DATA TO GPU LOADING
'''
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    