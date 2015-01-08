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


class Deep(object):
    
    def __init__(
        self,
        topology,
        nonlinearities,
        layer_types,
        device,
        regularisation,
        data,
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
        
        :type input_bias: boolean
        :param input_bias: NOT FINISHED
        
        '''
        self.topology = topology
        self.nonlinearities = nonlinearities
        self.layer_types = layer_types
        self.device = device
        self.regularisation = regularisation
        self.data = data
        
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
                    n_train_batches,
                    batch_size=10,
                    learning_rate=0.1,
                    pretrain_epochs=10,
                    corruption_level=0.2
                    ):
        '''
        Run through the layers of the network and load parameters
        '''
        for layer in self.net:
            layer.load_pretrain_params(
                loss_type,
                n_train_batches,
                batch_size=batch_size,
                learning_rate=learning_rate,
                pretrain_epochs=pretrain_epochs,
                corruption_level=corruption_level)
        
        print('Pretrain parameters loaded')
        


    def pretrain(self, optimisation_scheme='SGD'):
        
        ### CONSTRUCT EXPRESSION GRAPH ###
        
        #x = T.matrix(name='x', dtype=theano.config.floatX)
        
        start_time = time.clock()
        
        if self.device == 'AE':
            for i in np.arange(self.num_layers/2):
                print('Constructing expression graph for layer')
                layer = self.net[i]
                cost, updates = layer.get_cost_updates(learning_rate=layer.learning_rate)
                train_layer = theano.function([index],
                    cost,
                    updates=updates,
                    givens = {self.x: self.data.train_set_x[index * layer.batch_size: (index + 1) * layer.batch_size]})
                
                
                ### TRAIN ###
                for epoch in xrange(layer.pretrain_epochs):
                    # go through training set
                    c = []
                    for batch_index in xrange(layer.n_train_batches):
                        c.append(train_layer(batch_index))
                    
                    end_time = time.clock()
                    print('Layer %d, Training epoch %d, cost %5.3f, elapsed time %5.3f' \
                          % (i, epoch, np.mean(c), (end_time - start_time)))
            
            
            print('Pretraining complete: wrapping up')
            
            for i in np.arange(self.num_layers/2):
                
                layer = self.net[i]
                inverse_layer = self.net[self.num_layers-i-1]
                
                inverse_layer.W.set_value(layer.W.get_value().T)
                inverse_layer.b.set_value(layer.b2.get_value())


































































