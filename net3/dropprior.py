"""Autoencoder exhibiting the dropprior regulariser"""

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import sys

import cPickle
import gzip
import numpy
import theano.tensor as T
import theano.tensor.nnet as Tnet

from theano import config as Tconf
from theano import function as Tfunction
from theano import shared as TsharedX



class DroppriorTrain(Exception):

    def cost(self):
        Yhat = self.output
        Y = self.target
        loss = ((Y-Yhat)**2).sum(axis=1)
        return loss.mean()
    
    def updates(self, cost, params, args):
        '''Get parameter updates given cost'''
        learning_rate = args['learning_rate']
        updates = []
        for param in params:
            g_param = T.grad(cost, param)
            update = param - learning_rate * g_param
            updates.append((param, update))
        return updates
    
    def gpu_type(self, data):
        '''Convert data to theano.config.sharedX type'''
        data_typed = numpy.asarray(data, dtype=Tconf.floatX)
        dataX = TsharedX(data_typed, borrow=True)
        return dataX
    
    def load(self, data_address):
        '''Load the data from address'''
        f = gzip.open(data_address, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return (train_set, valid_set, test_set)
    
    def load2gpu(self, data_address):
        '''Load the data into the gpu'''
        train_set, valid_set, test_set = self.load(data_address)
        # Data should be stored columnwise
        self.train_x = self.gpu_type(train_set[0].T)
        self.train_y = self.gpu_type(train_set[1].T)
        self.valid_x = self.gpu_type(valid_set[0].T)
        self.valid_y = self.gpu_type(valid_set[1].T)
        self.test_x = self.gpu_type(test_set[0].T)
        self.test_y = self.gpu_type(test_set[1].T)
        self.num_train = train_set[0].shape[0]
        self.num_valid = valid_set[0].shape[0]
        self.num_test = test_set[0].shape[0]
        
    def build(self, args):
        '''Construct the model'''
        print('Building model')
        layer_sizes = args['layer_sizes']
        self.model = Dropprior(layer_sizes)
        self.input = T.matrix(name='input', dtype=Tconf.floatX)
        self.output = self.model.reconstruct(self.input)
        self.target = T.matrix(name='target', dtype=Tconf.floatX)
    
    def train(self, args):
        '''Train the model'''
        batch_size = args['batch_size']
        data_address = args['data_address']
        params = self.model._params
        print params
        print params[0].get_value().shape
        
        print('Loading data')
        self.load2gpu(data_address)
        
        print('Constructing flow graph')
        index = T.lscalar()
        
        cost = self.cost()
        updates = self.updates(cost, params, args)
        train_model = Tfunction(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: self.train_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.train_x[:,index*batch_size:(index+1)*batch_size]
            }
        )
        validate_model = Tfunction(
            inputs=[index],
            outputs=cost, ## CHANGE THIS LATER
            updates=updates,
            givens={
                self.input: self.valid_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.valid_x[:,index*batch_size:(index+1)*batch_size]
            }
        )
        
        print('Training')
        c = []
        num_train_batches = numpy.ceil(self.num_train / batch_size)
        for batch in numpy.arange(num_train_batches):
            c.append(train_model(batch))
        print numpy.asarray(c).mean()


class Dropprior(Exception):
    def __init__(self, layer_sizes):
        '''Construct the autoencoder expression graph'''

        self.ls = layer_sizes
        self.num_layers = len(self.ls) - 1
        
        self.W = []
        self.b = []
        self.c = []
        self._params = []
        for i in numpy.arange(self.num_layers):
            coeff = numpy.sqrt(6/(self.ls[i] + (self.ls[i+1])))
            W_value = coeff*numpy.random.uniform(size=(self.ls[i+1],self.ls[i]))
            W_value = numpy.asarray(W_value, dtype=Tconf.floatX)
            Wname = 'W' + str(i)
            self.W.append(TsharedX(W_value, Wname, borrow=True))
            
            b_value = 0.1*numpy.ones((self.ls[i+1],))
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True))
            
            c_value = 0.1*numpy.ones((self.ls[i],))
            c_value = numpy.asarray(c_value, dtype=Tconf.floatX)
            cname = 'c' + str(i)
            self.c.append(TsharedX(c_value, cname, borrow=True))
        
        for W, b, c in zip(self.W, self.b, self.c):
            self._params.append(W)
            self._params.append(b)
            #self._params.append(c)
    
    def encode_layer(self, X, layer):
        '''Sigmoid encoder function for single layer'''
        pre_act = T.dot(self.W[layer], X) + self.b[layer]
        return pre_act * (pre_act > 0)
    
    def decode_layer(self, h, layer):
        '''Linear decoder function for a single layer'''
        idx = self.num_layers - layer - 1
        pre_act = T.dot(self.W[idx].T, h) #+ self.c[idx]
        return pre_act * (pre_act > 0)
    
    def encode(self, X):
        '''Full encoder'''
        for i in numpy.arange(self.num_layers):
            X = self.encode_layer(X, i)
        return X
    
    def decode(self, h):
        '''Full decoder'''
        for i in numpy.arange(self.num_layers):
            h = self.decode_layer(h, i)
        return h

    def reconstruct(self, X):
        '''Reconstruct input'''
        h = self.encode(X)
        return self.decode(h)
    

if __name__ == '__main__':
    
    args = {
        'layer_sizes' : (784, 2000),
        'data_address' : './data/mnist.pkl.gz',
        'learning_rate' : 1e-4,
        'batch_size' : 100
        }
    
    dpt = DroppriorTrain()
    dpt.build(args)
    dpt.train(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    