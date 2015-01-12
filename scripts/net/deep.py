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
import time
import pickle


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
        np_rng=None,
        theano_rng=None,
        input_bias=None):
        

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
        
        :type input_bias: boolean
        :param input_bias: NOT FINISHED
        
        '''
        self.topology = topology
        self.nonlinearities = nonlinearities
        self.layer_types = layer_types
        self.device = device
        self.regularisation = regularisation
        self.data = data
        self.pkl_name = pkl_name
        
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

        # Build layers - deal with deafult inputs later
        self.x = T.matrix(name='x', dtype=theano.config.floatX)
        
        # The params object contains all network params
        self.params = []
        
        for i in np.arange(len(topology)-1):
            # Have ignored input_bias for now
            if i == 0:
                lyr = Layer(v_n=topology[i],
                            h_n=topology[i+1],
                            input=self.x,
                            layer_type=layer_types[i],
                            nonlinearity=nonlinearities[i],
                            h_reg=regularisation[0][0],
                            W_reg=regularisation[0][1],
                            np_rng=np_rng,
                            theano_rng=theano_rng,
                            W=None,
                            b=None,
                            b2=None,
                            mask=None)
                self.params.extend(lyr.params)
            else:
                lyr = Layer(v_n=topology[i],
                            h_n=topology[i+1],
                            input=self.net[i-1].output,
                            layer_type=layer_types[i],
                            nonlinearity=nonlinearities[i],
                            h_reg=regularisation[i][0],
                            W_reg=regularisation[i][1],
                            np_rng=np_rng,
                            theano_rng=theano_rng,
                            W=None,
                            b=None,
                            b2=None,
                            mask=None)
                self.params.extend(lyr.params)
            self.net.append(lyr)

        self.output = self.net[-1].output
        print('Network built')
        
    
    
    def initialise_weights(self, initialisation_regime):
        '''
        Run through the layers of the network and initialise one by one
        '''
        for layer in self.net:
            layer.init_weights(initialisation_regime=initialisation_regime, nonlinearity=layer.nonlinearity)
        
        print('Layers initialised')



    def load_pretrain_params(self,
                    loss_type,
                    optimisation_scheme,
                    layer_scheme,
                    n_train_batches,
                    batch_size=10,
                    pretrain_learning_rate=0.1,
                    pretrain_epochs=10,
                    corruption_level=0.2
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
                batch_size=batch_size,
                pretrain_learning_rate=pretrain_learning_rate,
                pretrain_epochs=pretrain_epochs,
                corruption_level=corruption_level)
        
        print('Pretrain parameters loaded')
        


    def pretrain(self):
        
        ### CONSTRUCT EXPRESSION GRAPH ###
        
        #x = T.matrix(name='x', dtype=theano.config.floatX)
        index = T.lscalar()  # index to a [mini]batch
        
        start_time = time.clock()
        
        if self.device == 'AE':
            
            print('Constructing expression graph for layers')
            
            pretrain_fns = []
            for i in np.arange(self.num_layers/2):
                layer = self.net[i]
                cost, updates = layer.get_cost_updates(learning_rate=layer.pretrain_learning_rate)
                train_layer = theano.function([index],
                    cost,
                    updates=updates,
                    givens = {self.x: self.data.train_set_x[index * layer.batch_size: (index + 1) * layer.batch_size]})
                pretrain_fns.append(train_layer)
                
            for i in np.arange(self.num_layers/2):
                ### TRAIN ###
                layer = self.net[i]
                for epoch in xrange(layer.pretrain_epochs):
                    # go through training set
                    c = []
                    for batch_index in xrange(layer.n_train_batches):
                        c.append(pretrain_fns[i](batch_index))
                    
                    end_time = time.clock()
                    print('Layer %d, Training epoch %d, cost %5.3f, elapsed time %5.3f' \
                          % (i, epoch, np.mean(c), (end_time - start_time)))
                
                self.pickle_machine(self.pkl_name)
            
            
            print('Pretraining complete: wrapping up')
            
            for i in np.arange(self.num_layers/2):
                
                layer = self.net[i]
                inverse_layer = self.net[self.num_layers-i-1]
                
                inverse_layer.W.set_value(layer.W.get_value().T)
                inverse_layer.b.set_value(layer.b2.get_value())
            
            # Now to rewrite the self.params variable to reflect the topological differences
            # between training and general inference
            self.params = []
            for layer in self.net:
                self.params.append(layer.W)
                self.params.append(layer.b)
                
            self.pickle_machine(self.pkl_name)
    
    
    def load_fine_tuning_params(self,
                                loss_type,
                                optimisation_scheme,
                                fine_tune_learning_rate,
                                max_epochs,
                                patience_increase,
                                n_train_batches,
                                n_valid_batches,
                                batch_size,
                                momentum,
                                tau=50,
                                pkl_rate=50):
        self.loss_type = loss_type
        self.optimisation_scheme = optimisation_scheme
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.max_epochs = max_epochs
        self.patience_increase = patience_increase
        self.n_train_batches = n_train_batches
        self.n_valid_batches = n_valid_batches
        self.batch_size = batch_size
        self.momentum = momentum
        self.tau=tau
        self.pkl_rate = pkl_rate
        
        # Some globally defined variables for synchronous updates
        self.epoch = 0
    
    
    def unsupervised_fine_tuning(self):
        '''
        Here we perform a Hinton-Salakhutdinov-esque fine-tuning run
        '''
        
        ### CONSTRUCT EXPRESSION GRAPH ###
        print('Constructing expression graph')
        
        index = T.lscalar()     # index to a [mini]batch

        self.cost, updates = self.get_cost_updates(learning_rate=self.fine_tune_learning_rate)
        train_all = theano.function([index],
            self.cost,
            updates=updates,
            givens = {self.x: self.data.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
        
        print('Fine_tuning')
        start_time = time.clock()
        
        best_valid_score = np.inf
        
        patience = 200
        done_looping = False
        
        while (self.epoch < self.max_epochs) and (not done_looping):
            self.epoch = self.epoch + 1
            
            c = []
            for batch_index in xrange(self.n_train_batches):
                c.append(train_all(batch_index))
            
            # Cross validate
            valid_score = self.cross_validate()
            valid_score = np.mean(valid_score)
            
            end_time = time.clock()
            print('Training epoch %d, cost %5.3f, elapsed time %5.3f' \
                  % (self.epoch, np.mean(c), (end_time - start_time)))
            
            
            # In future I wish to leverage the second GPU to perform validation
            if valid_score < best_valid_score:
                best_valid_score = valid_score
                # If we encounter a new best, increase patience
                patience = max(patience, self.epoch * self.patience_increase)
                
                print('     Best validation score: %5.3f, new patience: %d' % (best_valid_score, patience))
            
            if self.epoch >= patience:
                done_looping = True
                self.pickle_machine(self.pkl_name)
                
            if self.epoch % self.pkl_rate == 0:
                self.pickle_machine(self.pkl_name)

        
        
        
    
    def get_cost_updates(self, learning_rate):
        ### DEFINE COST FUNCTIONS AND UPDATES ###
        # For now we only use the standard SGD scheme
        z = self.net[-1].output
        
        if self.loss_type == 'L2':
            L = 0.5*T.sum((z - self.x)**2)
        
        cost = T.mean(L)
            
        # Gradient wrt parameters
        gparams = T.grad(cost, self.params)
        updates = []
        lr = learning_rate*self.get_learning_multiplier()
        
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - lr * ((1-self.momentum)*gparam + self.momentum*gparam)))
               
        return cost, updates
    
    
    
    def get_learning_multiplier(self):
        return self.tau/float(max(self.tau,self.epoch))


    
    def cross_validate(self):
        index = T.lscalar()
        
        valid_score = theano.function([index],
            self.cost,
            givens = {self.x: self.data.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
        
        return [valid_score(i) for i in xrange(self.n_valid_batches)]
    
    
    
    def pickle_machine(self, file_name):
        print('Pickling machine')
        stream = open(file_name,'w')
        
        # don't want to resave data
        data_copy = self.data
        self.data = []   
        pickle.dump(self, stream)
        
        self.data = data_copy




    def sample_AE(self, seed, num_samples, burn_in):
        '''
        The general idea is to Gibbs sample from the joint model implicitly
        defined by the AE by encoding, adding noise and then decoding iteratively.
        We should hopefully reach a period after burn_in where we are sampling
        from the true data distribution.
        
        One thing to consider for the future is the possibility of only breaking
        once and also having a stitch method to return the network to its original
        state before the break.
        
        :type seed: theano.confuig.floatX
        :param seed: matrix of sampler seeds
        
        :type num_samples: int
        :param num_samples: number of samples after burn-in
        
        :type burn_in: int
        :param burn_in: the burn-in duration
        '''
        position = len(self.topology)/2-1
        break_size = self.topology[position]
        
        # Define symbolic input
        input = T.matrix(name='input', dtype=theano.config.floatX)
        break_input = T.matrix(name='break_input', dtype=theano.config.floatX)
        
        # Break network
        self.break_output = self.break_network(position, break_input)
        self.net[0].switch_to_sample_mode()
        self.net[position+1].switch_to_sample_mode()
        
        # Define functions
        sample_encoder = theano.function([input], self.break_output)
        sample_decoder = theano.function([self.break_input], self.output)
        
        ### OKAY I HAVE TRIED TO IMPLEMNT TWO DIFFERENT THINGS AT THE SAME TIME AND
        # SHOULD NOW COME TO A DECISION. EITHER a) BREAK THE NETWORK AND INSERT A
        # CUSTOM SAMPLER b) DON'T BREAK THE NETWORK AND RELY ON THE AUTOMATIC
        # CORRUPTION IMPOSED BY Layer.switch_to_sample_mode()
        
        # Construct expression graph
        
        
        
        total_iter = num_samples + burn_in
        
        for i in xrange(total_iter):
            # Sample hidden representation
            
            
            # Sample visible data
            pass



    def break_network(self, position, break_input):
        '''
        Break the network into two distinct network objects. The interpretation
        of such an object may be to split an AE into encoder and decoder, such
        that we can inject noise for sampling.
        '''
        if position < 1:
            print('Break point too small')
            sys.exit(1)
        elif position > len(self.topology - 2):
            print('Break point too large')
            sys.exit(1)
        
        break_output = self.net[position].output
        self.net[position+1].input = break_input
        
        return break_output
        
        

    def get_corrupt(self, corruption_level):
            """ We use binary erasure noise """
            print('Corrupting test set')
            
            # Set up random number generators on CPU and GPU
            np_rng = np.random.RandomState(123)
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
            
            # Symbolic input
            input = T.dmatrix(name='input')
            
            # Define function
            corrupt = theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input
            
            # Construct expression graph
            fn = theano.function([input], corrupt)
            
            # Run function
            self.corrupt_set_x = theano.shared(np.asarray(fn(self.test_set_x.get_value()),
                                                          dtype=theano.config.floatX),
                                               borrow=True)
            
        





















































