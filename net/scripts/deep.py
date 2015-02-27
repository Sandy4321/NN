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
            
            
            
    def init_random_numbers(self, mode, shape):
        """
        Despite all the lovely things about theano the random number generation
        is one of frustrating sticking points. Not only are they harder to use
        but they are also very buggy. It appears that we need to create a numpy
        rng object and pass that through to a theano shared variable, which will
        presumeably be shipped to the GPU during graph construction. The various
        rng objects supplied by theano run at speeds differing by orders of magnitude
        so the solution found here is really just an empirical hack to speed things.
        """

        if not hasattr(self, 'rng'):
            self.rng = []
    
        if (mode == 'mask'):
            rng     = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            self.rng.append((rng,))
        elif (mode == 'bernoulli'):
            rng     = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            self.rng.append((rng,))
        elif (mode == 'salt_and_pepper'):
            rnga    = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            rngb    = theano.shared(np.asarray(self.np_rng.random_sample(shape)), \
                                      theano.config.floatX).astype(theano.config.floatX)
            self.rng.append((rnga,rngb))
        elif mode == 'gaussian':
            rng     = theano.shared(np.asarray(self.np_rng.randn(shape)), \
                                     theano.config.floatX).astype(theano.config.floatX)
            self.rng.append((rng,))
        else:
            print('Invalid noise type for initialisation')
            sys.exit(1)
        
        return len(self.rng) - 1
    
    

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
        


    def pretrain(self): 
        ### CONSTRUCT EXPRESSION GRAPH ###
        index               = T.lscalar()   # index to a [mini]batch
        iteration           = T.lscalar()   # iteration number
        num_pretrain_layers = self.num_layers
        if (self.device == 'AE') or (self.device == 'DAE'):
            num_pretrain_layers /= 2   
            
        print('Constructing expression graph for layers')
        pretrain_fns = []
        for i in np.arange(num_pretrain_layers):
            layer           = self.net[i]
            cost, updates   = layer.get_cost_updates(layer.pretrain_learning_rate)
            
            train_layer     = theano.function([index],
                cost,
                updates     = updates,
                givens      = {self.x: self.data.train_set_x[index * layer.batch_size: \
                                                             (index + 1) * layer.batch_size,:]})
            
            pretrain_fns.append(train_layer)
        
        print('Training')
        start_time  = time.clock()
        for i in np.arange(num_pretrain_layers):
            layer = self.net[i]
            for epoch in xrange(layer.pretrain_epochs):
                c = []
                
                self.net[0].set_iteration(epoch)
                for batch_index in xrange(layer.n_train_batches):
                    c.append(pretrain_fns[i](batch_index))
                end_time = time.clock()
               
                mc = np.mean(c)
                self.check_real(mc)     # Check for divergence
                print('Layer %d, Training epoch %d, cost %5.3f, elapsed time %5.3f' \
                      % (i, epoch, mc, (end_time - start_time)))
            self.pickle_machine(self.pkl_name)
            
        print('Pretraining complete: wrapping up')
        if (self.device == 'AE') or (self.device == 'DAE'):
            self.unfold_AE()
        self.set_params()

        self.pickle_machine(self.pkl_name)
    
    
    
    def check_real(self, x):
        if np.isnan(x):
            print('NaN error')
            raise DivergenceError('nan')
        elif np.isinf(x):
            print('INF error')
            raise DivergenceError('mc')
        
    
    
    def unfold_AE(self):
        num_pretrain_layers = self.num_layers/2
        for i in np.arange(num_pretrain_layers):
                layer = self.net[i]
                inverse_layer = self.net[self.num_layers-i-1] 
                inverse_layer.W.set_value(layer.W.get_value().T)
                inverse_layer.b.set_value(layer.b2.get_value())
                
                
    
    def set_params(self):
        self.params = []
        for layer in self.net:
            self.params.append(layer.W)
            self.params.append(layer.b)
    
            
    
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
        self.net[0].corruption_scheme = self.corruption_scheme

        self.cost, updates  	= self.get_cost_updates(learning_rate=self.fine_tune_learning_rate)
        train_all           	= theano.function([index],
            self.cost,
            updates 		= updates,
            givens  		= {self.x: self.data.train_set_x[index * self.batch_size: \
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
        ### DEFINE COST FUNCTIONS AND UPDATES ###
        # First we create a random number generator to corrupt the input, then we
	# define the corruption process, then we concatenate this process onto the
	# frontend of the NN. Finally we reference the output of this new NN in z. 

	if self.device == 'DAE':
            self.net[0].destroy_random_numbers()
            self.net[0].init_random_numbers(self.noise_type, (self.batch_size, self.topology[0]))
            x_tilde = self.net[0].get_corrupt(self.x, self.corruption_level)
            part_num = self.break_network(0, self.num_layers-1, x_tilde)
            print('Using fine-tune corruption')
        
        # For now we only use the standard SGD scheme
        z = self.part[part_num][2] 
	updates         = []
        self.velocities = []
        for param in self.params:
            self.velocities.append(theano.shared(np.zeros(param.get_value().shape, \
                                                   dtype=theano.config.floatX)))

        # LOSS
        if self.loss_type   == 'L2':
            L = 0.5*T.sum((z - self.x)**2, axis=1)
        elif self.loss_type == 'xent':
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        loss = T.mean(L)
        
        
        # REGULARISATION --- I would very much like to tidy this up at some point
        regularisation  = 0
        activation_grad = 0
        current_avg_h   = 0
        for i, layer in enumerate(self.net):
            # Weight decay
            if layer.W_reg == 'L1':
                regularisation += np.abs(layer.W.get_value(borrow=True)).sum()
            elif layer.W_reg == 'L2':
                regularisation += (layer.W.get_value(borrow=True)**2).sum()
            
            # Activation sparsity - apply to hidden neurons only
            if (layer.h_reg == 'KL') and (i < (self.num_layers - 1)) and (self.activation_weight != 0.0):
                current_avg_h += layer.output.sum()/self.num_h
   
        # Here we update the tracked hidden activation mean and compute the associated
        # activation cost gradient. Due to the non-self-evident relation of the average
        # hidden activation wrt the parameters, we hard code it in.
        if self.activation_weight != 0.0:
            self.avg_h      = self.h_track*self.avg_h + (1-self.h_track)*current_avg_h
            activation_grad = (self.avg_h - self.sparsity_target)/(self.avg_h*(1-self.avg_h) + 0.001)   # 0.001 regularisation
            act             = (self.activation_weight*activation_grad/self.training_size)
        else:
            act = 0
        act     = T.cast(act, dtype=theano.config.floatX)
        
        
        
        
        # COST = LOSS + REGULARISATION
        cost    = loss + (self.regularisation_weight*regularisation/self.training_size)
        
        # Gradient wrt parameters
        gparams = T.grad(cost, self.params)
        lr      = learning_rate*self.anneal()
        
        
        for param, gparam, velocity in zip(self.params, gparams, self.velocities):
            updates.append((velocity, self.momentum*velocity + lr*gparam*(1+act)))
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




    def sample_AE(self, seed, num_samples, noise_type, corruption_level):

        # Need to store seed device-side for GPU stuff to work and setup the
        # random number generators for MCMC if not already done
        #seed = theano.shared(seed, 'seed')
        self.init_corrupt()
        
        # Define some useful parameters for navigating the broken network
        end_position    = self.num_layers - 1
        break_position= (end_position + 1)/2 - 1
        #break_position  = end_position
        
        # Setting up the iterable sampling data structure
        seed_shape      = seed.shape
        seed_shape     += (num_samples+1,)    
        sample          = np.zeros(seed_shape, dtype=theano.config.floatX)
        sample[:,:,0]   = seed
        sample          = theano.shared(np.asarray(sample, dtype=theano.config.floatX))

        # Define input corruption process
        index = T.lscalar('index')
        pre_input       = T.matrix(name='pre_input', dtype=theano.config.floatX)
        rng_id          = self.init_random_numbers(noise_type, seed.shape)
        x_tilde         = self.get_corrupt(pre_input, noise_type, rng_id, corruption_level)
        
        # Concatenate the corruption process and encoder
        enc_idx         = self.break_network(0, break_position, x_tilde)
        
        # Define hidden layer sampling
        rng_id          = self.init_random_numbers('bernoulli', (seed.shape[0], self.topology[break_position+1]))
        h_stoc          = self.get_corrupt(self.part[enc_idx][2], 'bernoulli', rng_id)
        dec_idx         = self.break_network(break_position+1, end_position, h_stoc)
        
        # Need to work on a dict for the part labels
        sample_update   = (sample, T.set_subtensor(sample[:,:,index+1], self.part[dec_idx][2]))
        
        decrupt = theano.function([index],
            sb.gpu_from_host(self.part[dec_idx][2]),
            givens      = {pre_input: sample[:,:,index]},
            updates     = [sample_update])

        
        for i in xrange(num_samples):
            decrupt(i)
        
        return sample.get_value()           
        
        
        
    def get_corrupt(self, input, noise_type, rng_id, corruption_level=0.5):
        """
        Corrupt input
        """
        if noise_type == "mask":
            return (self.rng[rng_id][0]>corruption_level) * input
        if noise_type == "bernoulli":
            return (self.rng[rng_id][0]<input)
        elif noise_type == "salt_and_pepper":
            a = (self.rng[rng_id][0]>corruption_level)*1.0
            b = (self.rng[rng_id][1]>0.5)*1.0
            c = T.eq(a,0) * b
            return (input*a) + c
        elif noise_type == "gaussian":
            return self.rng[rng_id][0] * input
    


    def break_network(self, position_in, position_out, input):
        """
        Split the NN into two parts an completely reconstruct expression graphs
        
        :type position_in: int
        :param position_in: input layer
        
        :type position_in: int
        :param position_in: output layer
        
        :type input: theano.config.floatX
        :param input: data to enter network
        """
        
        # We break into an encoder and a decoder. First of all though we
        # need to check that the break position is valid
        assert position_in >= 0
        assert position_out < self.num_layers
        #assert position_in != position_out
        
        
        # Now we simply loop through the elements of 'topology' and create
        # layers for the new network copying from self
        
        net = []
        params = []
        num_layers = position_out + 1 - position_in
        
        for i in xrange(num_layers):
            
            position = position_in + i
            # Have ignored extra bells and whistles for now
            if i == 0:
                lyr = Layer(v_n         = self.topology[position],
                            h_n         = self.topology[position+1],
                            input       = input,
                            layer_type  = self.layer_types[position],
                            nonlinearity= self.nonlinearities[position],
                            h_reg       = self.regularisation[position][0],
                            W_reg       = self.regularisation[position][1],
                            np_rng      = self.np_rng,
                            theano_rng  = self.theano_rng,
                            W           = self.net[position].W,
                            b           = self.net[position].b,
                            b2          = None)
                params.extend(lyr.params)
            else:
                lyr = Layer(v_n         = self.topology[position],
                            h_n         = self.topology[position+1],
                            input       = net[i-1].output,
                            layer_type  = self.layer_types[position],
                            nonlinearity= self.nonlinearities[position],
                            h_reg       = self.regularisation[position][0],
                            W_reg       = self.regularisation[position][1],
                            np_rng      = self.np_rng,
                            theano_rng  = self.theano_rng,
                            W           = self.net[position].W,
                            b           = self.net[position].b,
                            b2          = None)
                params.extend(lyr.params)
            net.append(lyr)

        output = net[-1].output
        
        # We store the new network as an append to the self.part variable
        if hasattr(self, 'part'):
            self.part.append((net, params, output))
        else:
            self.part = []
            self.part.append((net, params, output))
        
        return len(self.part) - 1   # Returns index to part
        
        print('Network partition built')
        
        



















































