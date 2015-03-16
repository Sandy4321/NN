'''
The deep framework allows us to define objects with multiple stacked
layers and to train each one greedily, followed by a fine-tuning stage.
It is also possible to skip straight to the fine-tuning stage provided
measures have ben taken to ensure proper choice of nonlinearities and
regularisation etc.

@author: dew
@date: 8 Jan 2013
'''

from layer import Layer
from data_handling import Data_handling
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.cuda.basic_ops as sb
import time
import pickle
import sys

class DivergenceError(Exception): pass

class Deep(object):
    
    def __init__(
        self,
        topology,
        nonlinearities,
        layer_types,
        device,
        regularisation,
        data,
        pkl_name,
        np_rng		= None,
        theano_rng	= None,
        input_bias	= None):
        

        '''
        :type topology: tuple of ints
        :param topology: tuple defining NN topology
        
        :type nonlinearities: tuple of strings
        :param nonlinearities: tuple defining the nonlinearities of the network
        
        :type layer_types: tuple of strings
        :param layer_types: this defines the interpretation and training of the network
        
        :type device: string
        :param device: this specifies the kind of device we are using (useful for standard objects)
        
        :type regularisation: tuple of strings
        :type regularisation: this sets the regularisation of each layer ((h0,W0),(h1,W1),(h2,W2),...)
        
        :type data: Data_handling object
        :param data: this is the object where we have stored all training, validation and test data
        
        :type pkl_name: string
        :param pkl_name: name of file to pickle data to
        
	:type np_rng: numpy random number generator
	:param np_rng: the numpy random number generator object for DAES etc.

	:type theano_rng: theano random number generator 
	:param theano_rng: theano random number generator object for DAES etc
 
        :type input_bias: boolean
        :param input_bias: NOT FINISHED
        
        '''
        self.topology       = topology
        self.nonlinearities = nonlinearities
        self.layer_types    = layer_types
        self.device         = device
        self.regularisation = regularisation
        self.data           = data
        self.pkl_name       = pkl_name
        self.np_rng         = np_rng
        self.theano_rng     = theano_rng
        
        # Check random number generators
        self.init_corrupt()
        
        # We are going to store the layers of the NN in a net object, where
        # net[0] is the input layer etc. This has callable symbolic parameters
        # e.g. net[x].W, net[x].b
        
        self.net = []   
        
        # First we perform some sanity checks on the network coherency.
        
        # Make sure that the specfied nonlinearities correspond to layers,
        # otherwise throw an error and exit
        assert len(nonlinearities) == len(topology) - 1
        self.num_layers = len(nonlinearities)
       
        # Now we simply loop through the elements of 'topology' and create a
        # network layer with appropriate bound nonlinearities.

        # Build layers - deal with default inputs later
        self.x = T.matrix(name='x', dtype=theano.config.floatX)
        
        # The params object contains all network params
        self.params = []
        
        for i in np.arange(len(topology)-1):
            # Have ignored input_bias for now
            if i == 0:
                lyr = Layer(v_n         = topology[i],
                            h_n         = topology[i+1],
                            input       = self.x,
                            layer_type  = layer_types[i],
                            nonlinearity= nonlinearities[i],
                            h_reg       = regularisation[0][0],
                            W_reg       = regularisation[0][1],
                            np_rng      = self.np_rng,
                            theano_rng  = self.theano_rng,
                            W           = None,
                            b           = None,
                            b2          = None)
                self.params.extend(lyr.params)
            else:
                lyr = Layer(v_n         = topology[i],
                            h_n         = topology[i+1],
                            input       = self.net[i-1].output,
                            layer_type  = layer_types[i],
                            nonlinearity= nonlinearities[i],
                            h_reg       = regularisation[i][0],
                            W_reg       = regularisation[i][1],
                            np_rng      = self.np_rng,
                            theano_rng  = self.theano_rng,
                            W           = None,
                            b           = None,
                            b2          = None)
                self.params.extend(lyr.params)
            self.net.append(lyr)

        self.output = self.net[-1].output
        print('Network built')
        


    def init_corrupt(self):
        # Set up random number generators on CPU and GPU
        if self.np_rng is None:
            self.np_rng     = np.random.RandomState(123)
            
        if self.theano_rng is None:
            self.theano_rng = RandomStreams(self.np_rng.randint(2 ** 30))
            


    def load_pretrain_params(self,
                    loss_type,
                    optimisation_scheme,
                    layer_scheme,
                    n_train_batches,
                    batch_size              = 10,
                    pretrain_learning_rate  = 0.1,
                    pretrain_epochs         = 10,
                    initialisation_regime   = 'Glorot',
                    noise_type              = 'mask',
                    corruption_level        = 0.1,
                    corruption_scheme       = 'flat',
                    corruption_tau          = 1
                    ):
        '''
        Run through the layers of the network and load parameters
        '''
        for layer in self.net:
            layer.load_pretrain_params(
                loss_type,
                optimisation_scheme,
                layer_scheme,
                n_train_batches,
                batch_size                  = batch_size,
                pretrain_learning_rate      = pretrain_learning_rate,
                pretrain_epochs             = pretrain_epochs,
                initialisation_regime       = initialisation_regime,
                noise_type                  = noise_type,
                corruption_level            = corruption_level,
                corruption_scheme           = corruption_scheme,
                corruption_tau              = corruption_tau)
        
        print('Pretrain parameters loaded')
        
    
    
    def check_real(self, x):
        if np.isnan(x):
            print('NaN error')
            raise DivergenceError('nan')
        elif np.isinf(x):
            print('INF error')
            raise DivergenceError('mc')
        
            
    
    def load_fine_tuning_params(self,
                                loss_type,
                                optimisation_scheme,
                                fine_tune_learning_rate,
                                max_epochs,
                                validation_frequency,
                                patience_increase,
                                n_train_batches,
                                n_valid_batches,
                                batch_size              = 100,
                                momentum		= 0.99, 
				regularisation_weight   = 1e-6,
                                h_track                 = 0.995,
                                sparsity_target         = 0.05,
                                activation_weight       = 1e-6, 
                                tau                     = 50,
                                pkl_rate                = 50,
                                noise_type              = 'salt_and_pepper',
                                corruption_level        = 0.4,
                                corruption_scheme       = 'flat',
                                corruption_tau          = 1):
        self.loss_type                  = loss_type
        self.optimisation_scheme        = optimisation_scheme
        self.fine_tune_learning_rate    = fine_tune_learning_rate
        self.max_epochs                 = max_epochs
        self.validation_frequency       = validation_frequency
        self.patience_increase          = patience_increase
        self.n_train_batches            = n_train_batches
        self.n_valid_batches            = n_valid_batches
        self.batch_size                 = batch_size
        self.momentum                   = momentum
        self.regularisation_weight      = regularisation_weight
        self.h_track                    = h_track
        self.sparsity_target            = sparsity_target
        self.activation_weight          = activation_weight
        self.tau                        = tau
        self.pkl_rate                   = pkl_rate
        self.noise_type                 = noise_type
        self.corruption_level           = corruption_level
        self.corruption_scheme          = corruption_scheme
        self.corruption_tau             = corruption_tau
        
        # Some globally defined variables for synchronous updates
        self.epoch          = 0
        self.training_size  = batch_size*n_train_batches
        self.num_h          = 0
        for i in self.topology:
            self.num_h += i
        self.num_h -= (self.topology[0] + self.topology[-1])
        self.avg_h  = 0.0
    
    
    
    def unsupervised_fine_tuning(self):
        '''
        Here we perform a Hinton-Salakhutdinov-esque fine-tuning run with a Bengio twist.
        '''
        
        ### CONSTRUCT EXPRESSION GRAPH ###
        print('Constructing expression graph')
        
        index = T.lscalar()     # index to a [mini]batch
        #self.net[0].corruption_scheme = self.corruption_scheme

        self.cost, updates  	= self.get_cost_updates(learning_rate=self.fine_tune_learning_rate)
        train_all           	= theano.function([index],
            self.cost,
            updates 		= updates,
            givens  		= {self.x: self.data.train_set_x[index * self.batch_size: \
                                                                 (index + 1) * self.batch_size,:], \
				   self.label: self.data.train_set_y[index * self.batch_size: \
								 (index + 1) * self.batch_size,:]})
        
        print('Fine_tuning')
        start_time          	= time.clock()
        best_valid_score    	= np.inf
        patience            	= 200
        done_looping        	= False
        self.best_params    	= self.params
        
        while (self.epoch < self.max_epochs) and (not done_looping):
            self.epoch = self.epoch + 1
            c = []
            
            self.net[0].set_iteration(self.epoch)
            for batch_index in xrange(self.n_train_batches):
                c.append(train_all(batch_index))
            end_time    = time.clock()         
                            
            mc = np.mean(c)
            if np.isnan(mc):
                print('NaN error')
                raise DivergenceError('nan')
            elif np.isinf(mc):
                print('INF error')
                raise DivergenceError('mc')
                
            print('Training epoch %d, cost %5.3f, elapsed time %5.3f' \
                  % (self.epoch, mc, (end_time - start_time)))
            
            # Cross validate
            if self.epoch % self.validation_frequency == 0:
                valid_score = self.cross_validate()
                valid_score = np.mean(valid_score)
            
                # In future I wish to leverage the second GPU to perform validation
                if valid_score < best_valid_score:
                    best_valid_score = valid_score
                    # If we encounter a new best, increase patience
                    patience = max(patience, self.epoch * self.patience_increase)
                    print('     Best validation score: %5.3f, new patience: %d' % (best_valid_score, patience))
                    # And store the state of the system
                    self.best_params = self.params
                
            if self.epoch >= patience:
                done_looping = True
                break
                
            if self.epoch % self.pkl_rate == 0:
                self.pickle_machine(self.pkl_name)
        
        # Wrap up
        # Need to detach corruption from input if DAE
        if self.device == 'DAE':
            print('Network rebreak')
            self.break_network(0, self.num_layers-1, self.x)
            self.output = self.net[-1].output
            del self.part
        
        self.params = self.best_params
        del self.best_params
        self.pickle_machine(self.pkl_name)
        
        
    
    def get_cost_updates(self, learning_rate):
        '''
        First we create a random number generator to corrupt the input, then we
	define the corruption process, then we concatenate this process onto the
	frontend of the NN. Finally we reference the output of this new NN in z.
	'''
	

        # For now we only use the standard SGD scheme
	z 		= self.output
	self.label	= T.matrix(name='label', dtype=theano.config.floatX)
	updates         = []
        self.velocities = []
        for param in self.params:
            self.velocities.append(theano.shared(np.zeros(param.get_value().shape, \
                                                   dtype=theano.config.floatX)))

        # LOSS
        if self.loss_type   == 'L2':
            L	 = 0.5*T.sum((z - self.label)**2, axis=1)
        elif self.loss_type == 'xent':
            L 	= - T.sum(self.label * T.log(z) + (1 - self.label) * T.log(1 - z), axis=1)
        
        cost 	= T.mean(L)

        # Gradient wrt parameters
        gparams = T.grad(cost, self.params)
        lr      = learning_rate*self.anneal()
        
        
        for param, gparam, velocity in zip(self.params, gparams, self.velocities):
            updates.append((velocity, self.momentum*velocity + lr*gparam))
            updates.append((param, param - velocity))
        
        return cost, updates
    
    
    
    
    def anneal(self):
        return self.tau/float(max(self.tau,self.epoch))


    
    def cross_validate(self):
        index = T.lscalar()
        valid_score = theano.function([index],
            self.cost,
            givens = {self.x: self.data.valid_set_x[index * self.batch_size:(index + 1) * self.batch_size]})
        
        return [valid_score(i) for i in xrange(self.n_valid_batches)]
    
    
    
    def pickle_machine(self, file_name):
        print('Pickling machine')
        stream = open(file_name,'w')
        
        # don't want to resave data
        data_copy = self.data
        self.data = []   
        pickle.dump(self, stream, pickle.HIGHEST_PROTOCOL)
        
        self.data = data_copy



 
















































