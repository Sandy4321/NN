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
from mlp import Mlp, layer_from_sparsity, total_weights, write_neurons
from DGWN import Dgwn
from matplotlib import pylab
from matplotlib import pyplot as plt
from preprocess import Preprocess
from scipy.special import polygamma
from theano import config as Tconf
from theano.printing import Print as Tprint
from theano.tensor.extra_ops import to_one_hot
from theano import function as Tfunction
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared as TsharedX
from theano.tensor.shared_randomstreams import RandomStreams

class DivergenceError(Exception): pass

class Train():
    def __init__(self):
        self.epoch = 0
        self.pre = Preprocess()
        
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
        return (train_set, valid_set, test_set)
    
    def load2gpu(self, data_address, args):
        '''Load the data into the gpu'''
        learning_type = args['learning_type']
        binarize = args['binarize']
        train_set, valid_set, test_set = self.load(data_address)
        if binarize == True:
            train_set[0] = self.pre.binarize(train_set[0])
            valid_set[0] = self.pre.binarize(valid_set[0])
            test_set[0] = self.pre.binarize(test_set[0])
        self.num_train = train_set[0].shape[0]
        self.num_valid = valid_set[0].shape[0]
        self.num_test = test_set[0].shape[0] 
        # Shuffle data
        shuffle_train = numpy.random.permutation(self.num_train)
        shuffle_valid = numpy.random.permutation(self.num_valid)
        shuffle_test = numpy.random.permutation(self.num_test)
        train_set[0] = train_set[0][shuffle_train,:]
        train_set[1] = train_set[1][shuffle_train,:]
        valid_set[0] = valid_set[0][shuffle_valid,:]
        valid_set[1] = valid_set[1][shuffle_valid,:]
        test_set[0] = test_set[0][shuffle_test,:]
        test_set[1] = test_set[1][shuffle_test,:]
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
    
    def construct(self, model, args):
        '''Construct the model'''
        print('Constructing the model')
        self.model = model(args)
        
    def build(self, args):
        '''Connect inputs/outputs of the model'''
        self.input = T.matrix(name='input', dtype=Tconf.floatX)
        self.output, self.regularisation = self.model.predict(self.input, args)
        # Separate validation/test output is copy of train network with no dropout
        # NEED TO CHANGE
        test_args = args.copy()
        test_args['dropout_dict'] = None
        test_args['mode'] = 'validation'
        self.test_output, = self.model.predict(self.input, test_args)
        self.target = T.matrix(name='target', dtype=Tconf.floatX)
    
    def build_validate(self, args):
        '''Connect inputs/outputs of the model'''
        self.input = T.matrix(name='input', dtype=Tconf.floatX)
        self.test_output, = self.model.predict(self.input, args)
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
        # The learning rate scheduler is rather complicated, need to sort this
        self.learning_rate_margin = numpy.asarray(args['learning_rate_margin'])
        self.learning_rate_schedule = args['learning_rate_schedule']
        self.last_best = 0      # Last time we had a best cost
        self.lr_idx = 0
        self.last_idx = 0
        self.current_lr_idx = 0
        self.batch = 0
        # Variational dropout mask
        self.dropout_dict = args['dropout_dict']
 
        print('Constructing flow graph')
        index = T.lscalar()
        train_cost_type = args['train_cost_type']
        valid_cost_type = args['valid_cost_type']
        
        # CHANGE
        test_args = args.copy()
        test_args['dropout_dict'] = None
        test_args['mode'] = 'validation'
        
        # Costs and gradients
        train_cost, batch_train = self.cost(train_cost_type, self.target, self.output, args)
        valid_cost, batch_valid = self.cost(valid_cost_type, self.target, self.test_output, test_args)
        
        if args['cov'] == True:
            outputs = [valid_cost, self.model.X[1], self.model.XXT[1]]
        else:
            outputs = valid_cost
        
        updates = self.updates(train_cost, params, batch_train, args, self.regularisation)
        Tfuncs = {}
        Tfuncs['train_model'] = Tfunction(
            inputs=[index],
            outputs=[train_cost, self.regularisation],
            updates=updates,
            givens={
                self.input: self.train_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.train_y[:,index*batch_size:(index+1)*batch_size]
            }
        )    
        Tfuncs['validate_model'] = Tfunction(
            inputs=[index],
            outputs=outputs,
            givens={
                self.input: self.test_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.test_y[:,index*batch_size:(index+1)*batch_size]
            }
        )
        # Train
        monitor = self.main_loop(Tfuncs, args)
        return monitor
    
    def validate(self, args):
        '''Train the model'''
        index = T.lscalar()
        batch_size = int(args['batch_size'])
        params = self.model._params
        print('Constructing flow graph')
        valid_cost_type = args['valid_cost_type']       
        # Costs and gradients

        valid_cost, batch_valid = self.cost(valid_cost_type, self.target, self.test_output, args)
        Tfuncs = {}
        Tfuncs['validate_model'] = Tfunction(
            inputs=[index],
            outputs=valid_cost,
            givens={
                self.input: self.test_x[:,index*batch_size:(index+1)*batch_size],
                self.target: self.test_y[:,index*batch_size:(index+1)*batch_size]
            }
        )
        # Validate
        vc = self.run_once(Tfuncs, args)
        return vc
    
    def main_loop(self, Tfuncs, args):
        '''The main training loop'''
        train_model = Tfuncs['train_model']
        validate_model = Tfuncs['validate_model']
        batch_size = args['batch_size']
        num_epochs = int(args['num_epochs'])
        validation_freq = args['validation_freq']
        save_freq = args['save_freq']
        save_name = args['save_name']
        monitor = {'train_cost' : [], 'valid_cost' : [],
            'best_cost' : numpy.inf, 'best_model' : self.model._params,
            'XXT' : []}
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
            reg = numpy.zeros(num_train_batches)
            for self.batch in numpy.arange(num_train_batches):
                train_cost[self.batch], reg[self.batch] = train_model(self.batch)
            tc = (train_cost.sum()+reg.mean())/self.num_train
            train_cost += (reg/reg.shape[0])
            # I REALLY NEED TO RETHINK THE WAY HOW I DO THIS
            monitor['train_cost'].append(train_cost)
            print('%s%s Training cost: %f' % (ep, n(s), tc,)),
            self.check_real(tc)
            print('\t LR scaling: %0.3f' % (self.learning_rate_correction(),))
            
            # Validate
            if self.epoch % validation_freq == 0:
                valid_cost = numpy.zeros(num_valid_batches)
                if args['cov'] == True:
                    X = 0 
                    XXT = 0
                    Xtemp = 0
                    XXTtemp = 0
                for batch in numpy.arange(num_valid_batches):
                    if args['cov'] == True:
                        valid_cost[batch], Xtemp, XXTtemp = validate_model(batch)
                        X += Xtemp
                        XXT += XXTtemp
                    else:
                        valid_cost[batch] = validate_model(batch)
                vc = valid_cost.sum()/self.num_valid
                monitor['valid_cost'].append(vc)
                if args['cov'] == True:
                    XXT = XXT/self.num_valid - numpy.dot(X,X.T)/(self.num_valid**2)
                    monitor['XXT'].append(XXT)
                
                # Best model
                if args['valid_cost_type'] == 'accuracy':
                    if vc > monitor['best_cost']:
                        monitor['best_cost'] = vc
                        monitor['best_model'] = (self.model._params)
                        self.last_best = self.epoch
                        print('BEST MODEL: '),
                elif args['valid_cost_type'] != 'accuracy':
                    if vc < monitor['best_cost']:
                        monitor['best_cost'] = vc
                        monitor['best_model'] = (self.model._params)
                        self.last_best = self.epoch
                        print('BEST MODEL: '),
                print('%s%s Validation cost: %f' % (ep, n(s), vc,))
                self.check_real(vc)
            
            # Saving
            if self.epoch % save_freq == 0:
                self.save_state(save_name, args, monitor)
        
        self.save_state(save_name, args, monitor)
        return monitor
    
    def run_once(self, Tfuncs, args):
        '''The main training loop'''
        validate_model = Tfuncs['validate_model']
        batch_size = args['batch_size']
        save_name = args['save_name']
        num_valid_batches = numpy.ceil(self.num_valid / batch_size)
        # Validate
        valid_cost = numpy.zeros(num_valid_batches)
        for batch in numpy.arange(num_valid_batches):
            valid_cost[batch] = validate_model(batch)
        vc = valid_cost.sum()/self.num_valid
        print('Validation cost: %f' % (vc,))
        return vc
        
    def cost(self, cost_type, Y, Yhat, args):
        '''Evaluate the loss between prediction and target'''
        if 'num_samples' in args:
            ns = args['num_samples']
            if ns > 1:
                sh = Yhat.shape
                Yhat = T.reshape(Yhat,(sh[0],ns,sh[1]/ns))
                Yhat = T.mean(Yhat,axis=1)
        # Remember data is stored column-wise
        if cost_type == 'MSE':
            loss = 0.5*((Y-Yhat)**2).sum(axis=0)
        elif cost_type == 'cross_entropy':
            loss = Tnet.binary_crossentropy(Yhat, Y).sum(axis=0)
        elif cost_type == 'nll':       
            loss = -T.log((Y * Yhat).sum(axis=0))
        elif cost_type == 'accuracy':
            Yhatmax = T.argmax(Yhat, axis=0)
            Ymax = T.argmax(Y, axis=0)
            loss = T.sum(T.eq(Ymax,Yhatmax))
        else:
            print('Invalid cost')
            sys.exit(1)
        return (T.sum(loss), Y.shape[1])
    
    def updates(self, cost, params, batch_size, args, regularisation=0):
        '''Get parameter updates given cost'''
        # Load variables a check valid
        lr = args['learning_rate']*self.learning_rate_correction()
        lrm = args['lr_multipliers']
        assert lr >= 0
        momentum = args['momentum']
        ramp = args['momentum_ramp']
        assert (momentum >= 0 and momentum < 1) or (momentum == None)
        mmtm = self.momentum_ramp(0.5, momentum, ramp)
        cost = cost + regularisation*self.reg_mult(batch_size)

        # File updates
        updates = []
        # Standard parameter updates
        if args['algorithm'] == 'SGD':
            updates = self.SGD_updates(cost, params, args, lr, lrm, mmtm, updates)
        elif args['algorithm'] == 'NAG':
            updates = self.NAG_updates(cost, params, args, lr, lrm, mmtm, updates)     
        elif args['algorithm'] == 'RMSprop':
            updates = self.RMSprop_updates(cost, params, args, lr, lrm, updates)  
        elif args['algorithm'] == 'RMSNAG':
            updates = self.RMSNAG_updates(cost, params, args, lr, lrm, mmtm, updates)   
        else:
            print('Invalid training algorithm')
            sys.exit(1)
        self.max_norm(updates, args)
        return updates
    
    def reg_mult(self, batch_size):
        '''The regularisation multiplier'''
        M = self.num_train/(1.*batch_size)
        rm = 1./M
        return rm
    
    def momentum_ramp(self, start, end, ramp):
        '''Use a momentum ramp at the beginning'''
        if end != None:
            mult = numpy.minimum(self.epoch, ramp) / numpy.maximum(ramp*1., 1.)
            momentum = mult*end + (1-mult)*start
            momentum = momentum.astype(Tconf.floatX)
        else:
            momentum = None
        return momentum
    
    def lr_multipliers(self, lr_multipliers, param):
        '''Return the learning rate multipliers for each listed parameter'''
        lrm = 1.
        for key in lr_multipliers:
            if key in param.name:
                lrm = lr_multipliers[key]
                print('Param: %s \t LR Multiplier: %g' % (param, lrm))
        assert lrm >= 0, "Learning rate multiplier must be nonnegative"
        return lrm
    
    def SGD_updates(self, cost, params, args, lr, lrm, mmtm, updates):
        '''Stochastic gradient descent'''
        for param in params:
            lr = lr * self.lr_multipliers(lrm, param)
            g_param = T.grad(cost, param)
            # If no momentum set variable to None
            if mmtm != None:
                param_update = TsharedX(param.get_value()*0.,
                                        broadcastable=param.broadcastable)
                updates.append((param_update, mmtm*param_update - lr*g_param))
            else:
                param_update = lr*g_param
            updates.append((param, param + param_update))
        return updates
    
    def NAG_updates(self, cost, params, args, lr, lrm, mmtm, updates):
        '''Nesterov's Accelerated gradient'''
        for param in params:
                lr = lr * self.lr_multipliers(lrm, param)
                if mmtm == None:
                    print('For NAG require momentum in [0.,1.)')
                    sys.exit(1)
                velocity = TsharedX(param.get_value()*0.,
                                    broadcastable=param.broadcastable)
                # Updates
                g_param = T.grad(cost, param)
                # Update momentum
                updates.append((velocity, mmtm*velocity - lr*g_param))
                updates.append((param, param - lr*g_param + mmtm*velocity))
        return updates
    
    def RMSprop_updates(self, cost, params, args, lr, lrm, updates):
        '''Hinton's RMSprop (Coursera lecture 6)'''
        RMScoeff = args['RMScoeff']
        RMSreg = args['RMSreg']
        for param in params:
            lr = lr * self.lr_multipliers(lrm, param)
            g_param = T.grad(cost, param)
            ms = TsharedX(param.get_value()*0.,broadcastable=param.broadcastable)
            updates.append((ms, RMScoeff*ms + (1-RMScoeff)*(g_param**2)))
            param_update = lr*g_param/(T.sqrt(ms) + RMSreg)
            updates.append((param, param - param_update))
        return updates
    
    def RMSNAG_updates(self, cost, params, args, lr, lrm, mmtm, updates):
        '''RMSpropr with NAG'''
        RMScoeff = args['RMScoeff']
        RMSreg = args['RMSreg']
        for param in params:
            lr = lr * self.lr_multipliers(lrm, param)
            if mmtm == None:
                print('For NAG require momentum in [0.,1.)')
                sys.exit(1)
            velocity = TsharedX(param.get_value()*0.,
                                broadcastable=param.broadcastable)
            ms = TsharedX(param.get_value()*0.,
                                broadcastable=param.broadcastable)
            # Updates
            g_param = T.grad(cost, param)
            # Update RMS
            updates.append((ms, RMScoeff*ms + (1-RMScoeff)*(g_param**2)))
            param_update = g_param/(T.sqrt(ms) + RMSreg)
            # Update momentum
            updates.append((velocity, mmtm*velocity - lr*param_update))
            # Update parameters
            updates.append((param, param - lr*g_param + mmtm*velocity))
        return updates
    
    def max_norm(self, updates, args):
        '''Apply max norm constraint to the updates on weights'''
        max_row_norm = args['max_row_norm']
        if max_row_norm == None:
            pass
        else:         
            for i, update in enumerate(updates):
                param, param_update = update
                mn = False
                if hasattr(self.model, 'W'):
                    if param in self.model.W:
                        mn = True
                elif hasattr(self.model, 'M'):
                    if param in self.model.M:
                        mn = True
                if mn == True:
                    if args['norm'] == 'L2':
                        row_norms = T.sqrt(T.sum(T.sqr(param_update), axis=1, keepdims=True))
                    elif args['norm'] == 'Linf':
                        row_norms = T.max(T.abs_(param_update), axis=1, keepdims=True)
                    desired_norms = T.clip(row_norms, 0, max_row_norm)
                    constrained_W = param_update * (desired_norms / (1e-7 + row_norms))
                    # Tuples are immutable
                    updates_i = list(updates[i])
                    updates_i[1] = constrained_W.astype(Tconf.floatX)
                    updates[i] = tuple(updates_i)

    def learning_rate_correction(self):
        '''Learning rate schedule - COMPLICATED, NEED TO SORT OUT'''
        # Get learning rate tuple for this epoch
        idx = [x[0] for x in enumerate(self.learning_rate_margin) if x[1] <= self.epoch]
        idx = numpy.amax(idx)
        lr_idx_temp = self.lr_idx
        # Has learning rate tuple changed?
        if self.last_idx != idx:
            # Yes
            self.lr_idx = 0
            self.current_lr_idx = 0
        else:
            # No - how long since last best cost?
            best_marg = self.epoch - self.last_best
            # Only go to next element in tuple when best_marg >= 40 AND been at
            # current rate for >= 40
            if (best_marg >= 40) and (self.current_lr_idx >= 40):
                lr_idx_temp = numpy.minimum(self.lr_idx + 1, len(self.learning_rate_schedule[idx])-1)
        lrm = self.learning_rate_schedule[idx][lr_idx_temp]
        # Update caches
        self.last_idx = idx
        if self.lr_idx == lr_idx_temp:
            self.current_lr_idx += 1
        else:
            self.current_lr_idx = 0
        self.lr_idx = lr_idx_temp
        return lrm
    
    def check_real(self, x):
        '''Check training has not diverged'''
        if numpy.isnan(x):
            print('NaN error')
            raise DivergenceError('nan')
        elif numpy.isinf(x):
            print('INF error')
            raise DivergenceError('mc')
    
    def load_state(self, model, address):
        '''Load data from a pkl file'''
        print('Loading file')
        fp = open(address, 'r')
        state = cPickle.load(fp)
        args = state['args']
        monitor = state['monitor']
        fp.close()
        params = monitor['best_model']
        self.model = model(args)
        self.model.load_params(params, args)
    
    def save_state(self, fname, args, monitor=None):
        '''Save data to a pkl file'''
        print('Pickling: %s' % (fname,))
        state = {'args' : args,
                 'monitor' : monitor}
        stream = open(fname,'w')
        cPickle.dump(state, stream, protocol=cPickle.HIGHEST_PROTOCOL)
        stream.close()
    
    def log_stats(self):
        '''Make copies of the parameter statistics'''
        pass
    
    def extra_samples(self, X, args):
        '''Make parallel copies of the data'''
        mode = args['mode']
        n = args['num_samples']
        Y = T.concatenate([X,]*args['num_samples'], axis=1)
        return Y

























