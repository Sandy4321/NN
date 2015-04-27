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

from autoencoder import Autoencoder
from mlp_dev import Mlp
from matplotlib import pylab
from matplotlib import pyplot as plt
from theano import config as Tconf
from theano.tensor.extra_ops import to_one_hot
from theano import function as Tfunction
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams

class DivergenceError(Exception): pass

class Train():
    def __init__(self):
        self.epoch = 0
        
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
        train_set = list(train_set)
        valid_set = list(valid_set)
        test_set = list(test_set)
        train_set[1] = train_set[1][:,numpy.newaxis]
        valid_set[1] = valid_set[1][:,numpy.newaxis]
        test_set[1] = test_set[1][:,numpy.newaxis]
        train_set = tuple(train_set)
        valid_set = tuple(valid_set)
        test_set = tuple(test_set)
        return (train_set, valid_set, test_set)
    
    def load2gpu(self, data_address, args):
        '''Load the data into the gpu'''
        learning_type = args['learning_type']
        
        train_set, valid_set, test_set = self.load(data_address)
        # Data should be stored columnwise
        self.train_x = self.gpu_type(train_set[0].T)
        self.valid_x = self.gpu_type(valid_set[0].T)
        self.test_x = self.gpu_type(test_set[0].T)
        
        if learning_type == 'supervised':
            self.train_y = self.gpu_type(train_set[1].T)
            self.valid_y = self.gpu_type(valid_set[1].T)
            self.test_y = self.gpu_type(test_set[1].T)
        elif learning_type == 'unsupervised':
            self.train_y = self.train_x
            self.valid_y = self.valid_x
            self.test_y = self.test_x
        elif learning_type == 'classification':
            nc = args['num_classes']
            # Need to convert the target values into one-hot encodings
            train_y = numpy.zeros((train_set[1].shape[0], nc), dtype=Tconf.floatX)
            valid_y = numpy.zeros((valid_set[1].shape[0], nc), dtype=Tconf.floatX)
            test_y = numpy.zeros((test_set[1].shape[0], nc), dtype=Tconf.floatX)
            
            train_y[numpy.arange(train_y.shape[0]), train_set[1][:,0]] = 1
            valid_y[numpy.arange(valid_y.shape[0]), valid_set[1][:,0]] = 1
            test_y[numpy.arange(test_y.shape[0]), test_set[1][:,0]] = 1

            self.train_y = self.gpu_type(train_y.T)
            self.valid_y = self.gpu_type(valid_y.T)
            self.test_y = self.gpu_type(test_y.T)
        else:
            print('Invalid learning type')
            sys.exit(1)
        
        self.num_train = train_set[0].shape[0]
        self.num_valid = valid_set[0].shape[0]
        self.num_test = test_set[0].shape[0]          
        
    def build(self, model, args):
        '''Construct the model'''
        print('Building model')
        self.model = model(args)
        self.input = T.matrix(name='input', dtype=Tconf.floatX)
        self.output = self.model.predict(self.input, args)
        self.target = T.matrix(name='target', dtype=Tconf.floatX)
    
    def load_data(self, args):
        '''Load the data'''
        data_address = args['data_address']
        print('Loading data')
        self.load2gpu(data_address, args)
    
    def train(self, args):
        '''Train the model'''
        batch_size = int(args['batch_size'])
        params = self.model._params
        self.learning_rate_margin = args['learning_rate_margin']
        
        print('Constructing flow graph')
        index = T.lscalar()
        train_cost_type = args['train_cost_type']
        valid_cost_type = args['valid_cost_type']
        
        train_cost = self.cost(train_cost_type)
        valid_cost = self.cost(valid_cost_type)
        
        updates = self.updates(train_cost, params, args)
        Tfuncs = {}
        Tfuncs['train_model'] = Tfunction(
            inputs=[index],
            outputs=train_cost,
            updates=updates,
            givens={
                self.input: self.train_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.train_y[:,index*batch_size:(index+1)*batch_size]
            }
        )
        Tfuncs['validate_model'] = Tfunction(
            inputs=[index],
            outputs=valid_cost,
            givens={
                self.input: self.valid_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.valid_y[:,index*batch_size:(index+1)*batch_size]
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
        if args['valid_cost_type'] == 'accuracy':
            monitor['best_cost'] = -numpy.inf
        
        print('Training')
        num_train_batches = numpy.ceil(self.num_train / batch_size)
        num_valid_batches = numpy.ceil(self.num_valid / batch_size)
        # Timing, displays and monitoring
        s = self.set_start()
        n = self.now
        for self.epoch in numpy.arange(num_epochs):
            ep = '[EPOCH %05d]' % (self.epoch,)
            
            # Train
            train_cost = numpy.zeros(num_train_batches)
            for batch in numpy.arange(num_train_batches):
                train_cost[batch] = train_model(batch)
            tc = train_cost.mean()
            monitor['train_cost'].append(tc)
            print('%s%s Training cost: %f' % (ep, n(s), tc,))
            self.check_real(tc)
            
            # Validate
            if self.epoch % validation_freq == 0:
                valid_cost = numpy.zeros(num_valid_batches)
                for batch in numpy.arange(num_valid_batches):
                    valid_cost[batch] = validate_model(batch) 
                vc = valid_cost.mean()
                monitor['valid_cost'].append(vc)
                
                # Best model
                if args['valid_cost_type'] == 'accuracy':
                    if vc > monitor['best_cost']:
                        monitor['best_cost'] = vc
                        monitor['best_model'] = self.model._params
                        print('BEST MODEL: '),
                elif args['valid_cost_type'] != 'accuracy':
                    if vc < monitor['best_cost']:
                        monitor['best_cost'] = vc
                        monitor['best_model'] = self.model._params
                        print('BEST MODEL: '),
                print('%s%s Validation cost: %f' % (ep, n(s), vc,))
                self.check_real(vc)
            
            # Saving
            if self.epoch % save_freq == 0:
                self.save_state(save_name, args, monitor)
        
        self.save_state(save_name, args, monitor)
        return monitor
        
    def cost(self, cost_type):
        '''Evaluate the loss between prediction and target'''
        Yhat = self.output
        Y = self.target
        if cost_type == 'MSE':
            loss = 0.5*((Y-Yhat)**2).sum(axis=0)
        elif cost_type == 'cross_entropy':
            loss = Tnet.binary_crossentropy(Yhat, Y).sum(axis=0)
        elif cost_type == 'nll':
            loss = -T.log((Y * Yhat).sum(axis=0))
        elif cost_type == 'accuracy':
            Yhatmax = T.argmax(Yhat, axis=0)
            Ymax = T.argmax(Y, axis=0)
            loss = T.mean(T.eq(Ymax,Yhatmax))
        else:
            print('Invalid cost')
            sys.exit(1)
        return loss.mean()
    
    def updates(self, cost, params, args):
        '''Get parameter updates given cost'''
        # Load variables a check valid
        lr = args['learning_rate']*self.learning_rate_correction()
        assert lr >= 0
        mmtm = args['momentum']
        assert (mmtm >= 0 and mmtm < 1) or (mmtm == None)
        
        # File updates
        updates = []
        # Standard SGD with momentum
        if args['algorithm'] == 'SGD':
            for param in params:
                g_param = T.grad(cost, param)
                # If no momentum set variable to None
                if mmtm != None:
                    param_update = TsharedX(param.get_value()*0.,
                                            broadcastable=param.broadcastable)
                    updates.append((param_update, mmtm*param_update + lr*g_param))
                else:
                    param_update = lr*g_param
                updates.append((param, param - param_update))
        # Hinton's RMSprop (Coursera lecture 6) without Nesterov momentum            
        elif args['algorithm'] == 'RMSprop':
            RMScoeff = args['RMScoeff']
            RMSreg = args['RMSreg']
            for param in params:
                g_param = T.grad(cost, param)
                ms = TsharedX(param.get_value()*0.,
                                            broadcastable=param.broadcastable)
                updates.append((ms, RMScoeff*ms + (1-RMScoeff)*(g_param**2)))
                param_update = lr*g_param/(T.sqrt(ms) + RMSreg)
                updates.append((param, param - param_update))
        elif args['algorithm'] == 'SGSD':
            for param in params:
                g_param = T.sgn(T.grad(cost, param))
                # If no momentum set variable to None
                if mmtm != None:
                    param_update = TsharedX(param.get_value()*0.,
                                            broadcastable=param.broadcastable)
                    updates.append((param_update, mmtm*param_update + lr*g_param))
                else:
                    param_update = lr*g_param
                updates.append((param, param - param_update))
        else:
            print('Invalid training algorithm')
            sys.exit(1)
        
        self.max_norm(updates, args)
        
        return updates
    
    def max_norm(self, updates, args):
        '''Apply max norm constraint to the updates on weights'''
        max_row_norm = args['max_row_norm']
        if max_row_norm == None:
            pass
        else:         
            norm = args['norm']
            if norm == 'L2':  
                for i, update in enumerate(updates):
                    param, param_update = update
                    if param in self.model.W:
                        row_norms = T.sqrt(T.sum(T.sqr(param_update), axis=1, keepdims=True))
                        desired_norms = T.clip(row_norms, 0, max_row_norm)
                        constrained_W = param_update * (desired_norms / (1e-7 + row_norms))
                        # Tuples are immutable
                        updates_i = list(updates[i])
                        updates_i[1] = constrained_W
                        updates[i] = tuple(updates_i)
            elif norm == 'Linf':
                for i, update in enumerate(updates):
                    param, param_update = update
                    if param in self.model.W:
                        row_norms = T.max(T.abs_(param_update), axis=1, keepdims=True)
                        desired_norms = T.clip(row_norms, 0, max_row_norm)
                        constrained_W = param_update * (desired_norms / (1e-7 + row_norms))
                        # Tuples are immutable
                        updates_i = list(updates[i])
                        updates_i[1] = constrained_W
                        updates[i] = tuple(updates_i)
    
    def learning_rate_correction(self):
        tau = self.learning_rate_margin*1.
        return tau/float(max(tau, self.epoch))
    
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
        state = {'hyperparams' : args,
                 'monitor' : monitor}
        stream = open(fname,'w')
        cPickle.dump(state, stream, protocol=cPickle.HIGHEST_PROTOCOL)
        stream.close()
    
    def log_stats(self):
        '''Make copies of the parameter statistics'''
        pass
    

if __name__ == '__main__':
        
    args = {
        'algorithm' : 'SGD',
        'learning_type' : 'classification',
        'num_classes' : 10,
        'train_cost_type' : 'nll',
        'valid_cost_type' : 'accuracy',
        'layer_sizes' : (784, 1000, 1000, 1000, 10),
        'nonlinearities' : ('ReLU', 'ReLU', 'ReLU', 'SoftMax'),
        'data_address' : './data/mnist.pkl.gz',
        'learning_rate' : 1e-6,
        'learning_rate_margin' : 34,
        'momentum' : 0.85,
        'batch_size' : 100,
        'num_epochs' : 1000,
        'norm' : 'L2',
        'max_row_norm' : 3.5,
        'dropout_dict' : None, 
        'validation_freq' : 5,
        'save_freq' : 5,
        'save_name' : 'train_dev.pkl'
        }
    
    dropout_dict = {}
    for i in numpy.arange(len(args['nonlinearities'])):
        name = 'layer' + str(i)

        if i == 0:
            # Need to cast to floatX or the computation gets pushed to the CPU
            v = 0.8*numpy.ones((784,1)).astype(Tconf.floatX)
        else:
            #v = numpy.random.beta(a, b, size=(2000,1)).astype(Tconf.floatX)
            v = 0.5*numpy.ones((args['layer_sizes'][i],1)).astype(Tconf.floatX)
        sub_dict = { name : {'seed' : 234,
                             'type' : 'unbiased',
                             'values' : v}}
        dropout_dict.update(sub_dict)
    args['dropout_dict'] = dropout_dict
    #numpy.random.seed(seed=234)
    
    tr = Train()
    tr.build(Mlp, args)
    tr.load_data(args)
    monitor = tr.train(args)
    
    '''
    TODO: PRETRAINING, CONVOLUTIONS
    ''' 
