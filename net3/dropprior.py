"""Autoencoder exhibiting the dropprior regulariser"""

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

from matplotlib import pyplot as plt
from theano import config as Tconf
from theano import function as Tfunction
from theano import shared as TsharedX

class DivergenceError(Exception): pass

class Train():
    # Some handy timing functions
    def set_start(self):
        '''Set timer to zero'''
        s = time.time()
        return s
    
    def now(self, s):
        '''Readout time elapsed'''
        secs = time.time() - s
        strsecs = '%06.1f' % (secs,)
        message = '[TIME: ' + strsecs + ']'
        return message
    
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
        
    def build(self, model, args):
        '''Construct the model'''
        print('Building model')
        layer_sizes = args['layer_sizes']
        self.model = model(layer_sizes)
        self.input = T.matrix(name='input', dtype=Tconf.floatX)
        self.output = self.model.reconstruct(self.input)
        self.target = T.matrix(name='target', dtype=Tconf.floatX)
    
    def train(self, args):
        '''Train the model'''
        data_address = args['data_address']
        batch_size = int(args['batch_size'])
        params = self.model._params
        
        print('Loading data')
        self.load2gpu(data_address)
        
        print('Constructing flow graph')
        index = T.lscalar()
        cost = self.cost()
        updates = self.updates(cost, params, args)
        Tfuncs = {}
        Tfuncs['train_model'] = Tfunction(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: self.train_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.train_x[:,index*batch_size:(index+1)*batch_size]
            }
        )
        Tfuncs['validate_model'] = Tfunction(
            inputs=[index],
            outputs=cost, ## CHANGE THIS LATER
            givens={
                self.input: self.valid_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.valid_x[:,index*batch_size:(index+1)*batch_size]
            }
        )
        # Train
        monitor = self.main_loop(Tfuncs, args)
        return monitor
    
    def main_loop(self, Tfuncs, args):
        '''The main training loop'''
        train_model = Tfuncs['train_model']
        validate_model = Tfuncs['validate_model']
        batch_size = args['batch_size']
        num_epochs = int(args['num_epochs'])
        validation_freq = args['validation_freq']
        save_freq = args['save_freq']
        save_name = args['save_name']
        monitor = {'train_cost' : [],
                'valid_cost' : [],
                'best_cost' : numpy.inf,
                'best_model' : self.model._params}
        
        print('Training')
        num_train_batches = numpy.ceil(self.num_train / batch_size)
        num_valid_batches = numpy.ceil(self.num_valid / batch_size)
        # Timing, displays and monitoring
        s = self.set_start()
        n = self.now
        for epoch in numpy.arange(num_epochs):
            ep = '[EPOCH %05d]' % (epoch,)
            
            # Train
            train_cost = numpy.zeros(num_train_batches)
            for batch in numpy.arange(num_train_batches):
                train_cost[batch] = train_model(batch)
            tc = train_cost.mean()
            monitor['train_cost'].append(tc)
            print('%s%s Training cost: %f' % (ep, n(s), tc,))
            self.check_real(tc)
            
            # Validate
            if epoch % validation_freq == 0:
                valid_cost = numpy.zeros(num_valid_batches)
                for batch in numpy.arange(num_valid_batches):
                    valid_cost[batch] = validate_model(batch) 
                vc = valid_cost.mean()
                monitor['valid_cost'].append(vc)
                
                # Best model
                if vc < monitor['best_cost']:
                    monitor['best_cost'] = vc
                    monitor['best_model'] = self.model._params
                    print('BEST MODEL: '),
                print('%s%s Validation cost: %f' % (ep, n(s), vc,))
                self.check_real(vc)
            
            # Saving
            if epoch % save_freq == 0:
                self.save_state(save_name, args, monitor)
        
        self.save_state(save_name, args, monitor)
        return monitor
        
    def cost(self):
        Yhat = self.output
        Y = self.target
        loss = ((Y-Yhat)**2).sum(axis=1)
        return loss.mean()
    
    def updates(self, cost, params, args):
        '''Get parameter updates given cost'''
        # Load variables a check valid
        lr = args['learning_rate']
        assert lr >= 0
        mmtm = args['momentum']
        assert (mmtm >= 0 and mmtm < 1) or (mmtm == None)
        
        # File updates
        updates = []
        for param in params:
            g_param = T.grad(cost, param)
            # If no momentum set variable to None
            if mmtm != None:
                param_update = TsharedX(param.get_value()*0., broadcastable=param.broadcastable)
                updates.append((param_update, mmtm*param_update + lr*g_param))
            else:
                param_update = lr*g_param
            updates.append((param, param - param_update))
        return updates
    
    def check_real(self, x):
        '''Check training has not diverged'''
        if numpy.isnan(x):
            print('NaN error')
            raise DivergenceError('nan')
        elif numpy.isinf(x):
            print('INF error')
            raise DivergenceError('mc')
    
    def load_state(self):
        '''Load data from a pkl file'''
        pass
    
    def save_state(self, fname, args, monitor=None):
        '''Save data to a pkl file'''
        print('Pickling: %s' % (fname,))
        state = {'params' : self.model._params,
                 'hyperparams' : args,
                 'monitor' : monitor}
        stream = open(fname,'w')
        cPickle.dump(state, stream, protocol=cPickle.HIGHEST_PROTOCOL)
        stream.close()
    
    def log_stats(self):
        '''Make copies of the parameter statistics'''
        pass
    

class Autoencoder():
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
            
            b_value = 0.1*numpy.ones((self.ls[i+1],))[:,numpy.newaxis]
            b_value = numpy.asarray(b_value, dtype=Tconf.floatX)
            bname = 'b' + str(i)
            self.b.append(TsharedX(b_value, bname, borrow=True, broadcastable=(False,True)))
            
            c_value = 0.1*numpy.ones((self.ls[i],))[:,numpy.newaxis]
            c_value = numpy.asarray(c_value, dtype=Tconf.floatX)
            cname = 'c' + str(i)
            self.c.append(TsharedX(c_value, cname, borrow=True, broadcastable=(False,True)))
        
        for W, b, c in zip(self.W, self.b, self.c):
            self._params.append(W)
            self._params.append(b)
            self._params.append(c)
    
    def encode_layer(self, X, layer):
        '''Sigmoid encoder function for single layer'''
        pre_act = T.dot(self.W[layer], X) + self.b[layer]
        return pre_act * (pre_act > 0)
    
    def decode_layer(self, h, layer):
        '''Linear decoder function for a single layer'''
        idx = self.num_layers - layer - 1
        pre_act = T.dot(self.W[idx].T, h) + self.c[idx]
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
        'layer_sizes' : (784, 2000, 2000),
        'data_address' : './data/mnist.pkl.gz',
        'learning_rate' : 1e-4,
        'momentum' : 0.9,
        'batch_size' : 100,
        'validation_freq' : 5,
        'save_freq' : 5,
        'save_name' : 'dropprior.pkl'
        }
    
    tr = Train()
    model = Autoencoder
    tr.build(model, args)
    monitor = tr.train(args)
    '''
    fig = plt.figure()
    plt.plot(monitor['train_cost'])
    plt.show()
    '''
    
    '''
    TODO: EXCEPTIONS, DROPOUT, WEIGHT CONSTRAINTS, PRETRAINING, CONVOLUTIONS
    
    A problem with dropout is in deciding whether it is part of the model or
    the optimisation. I am going to side with the view that an optimisation is
    independent of the model (as much as possible) and has the sole aim of
    reaching a local minimum. Apart from early stopping, regularised SGD is
    actually SGD on a regularised objective, where the regularised objective
    is the model instead of an optimisation trick.
    ''' 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
