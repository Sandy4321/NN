'''
Script to train a a neural network
'''
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tnet
import sys

class Train:
    
    def train_layer_build(self, NN, layer_number):
        '''
        This section is where we define all of the callable functions, such as
        the cost etc.
        '''
        
        layer = NN.net[layer_number]
        
        if layer.method == "AE":
            # Here we train a layer as if it were an autoencoder. In fact we train
            # layers in pairs. We don't actually summon the decoder layer, we merely
            # hallucinate it and then copy over the parameters when done training
            # the current layer.
            
            if (layer_number >= NN.num_layers/2):
                layer.skip_layer = 'skip'
                print("Skipped decoder layers of AE")
            else:
                layer.skip_layer = 'no_skip'
                layer.W_prime = layer.W.T
                layer.b_prime = NN.net[NN.num_layers - layer_number - 1].b
                
                # Define cost
                a = T.vector('a',dtype=theano.config.floatX)
                b = T.vector('b',dtype=theano.config.floatX)
                if layer.loss == 'SE':
                    L = T.mean(0.5 * T.sum((a - b)**2, axis=1))
                elif layer.loss == 'xent':
                    L = - T.mean(T.sum(a * T.log(b) + (1 - a) * T.log(1 - b), axis=1))
                layer.loss_fn = theano.function([a, b], L)
                
                # Bind a reconstruction layer (inverse_layer) with appropriate
                # nonlinearity for greedy pre-training
                layer.inverse_layer = NN.net[NN.num_layers - layer_number - 1]
                
               # x = T.vector('x',dtype=theano.config.floatX)
              #  y = layer.get_output(x)
               # z = inverse_layer.get_output(y, layer.W_prime) 
                #recon = theano.function([x, newW], z, givens=[(layer.W_prime, newW)])
 
                # Define extra levels of regularisation - needs work!!!
                #if layer.regulariser == 'L2':
                #    regulariser = 0.0001 * (layer.W**2).sum()
                

                
                print("Built pretrain for layer %d using AE method" % layer_number)
                        
        else:
            print("Pretraining method not recognised")
            sys.exit(1)
            

            
    
    def train_layer(self, NN, layer_number):
        '''
        The layerwise training method contains (/will contain) a variety of
        algorithms to train the parameters of a single layer of the neural network,
        so as to minimise a particular loss function.
        
        Supported methods:
            - autoencoder (AE) training
            - denoising AE training
            - restricted boltzmann machine training
            - custom training
        '''
        layer = NN.net[layer_number]
        
        if layer.skip_layer == 'skip':
            pass
        elif layer.skip_layer == 'no_skip':
            
            # Where the magic happens
            '''
             print '... getting the pretraining functions'
            pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                        batch_size=batch_size)
        
            print '... pre-training the model'
            start_time = time.clock()
            ## Pre-train layer-wise
            corruption_levels = [.1, .2, .3]
            for i in xrange(sda.n_layers):
                # go through pretraining epochs
                for epoch in xrange(pretraining_epochs):
                    # go through the training set
                    c = []
                    for batch_index in xrange(n_train_batches):
                        c.append(pretraining_fns[i](index=batch_index,
                                 corruption=corruption_levels[i],
                                 lr=pretrain_lr))
                    print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                    print numpy.mean(c)
        
            end_time = time.clock()
        
            print >> sys.stderr, ('The pretraining code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))
            # end-snippet-4
            '''
            
            
            
            
            
            
            
            
            
        else:
            print "Skip layer parameter no specified"
            sys.exit(1)
        
            
            # Define optimisation technique
            
            
            #lyr.loss = loss
            #lyr.regulariser = regulariser
            #lyr.optimiser = optimiser
            #lyr.momentum = momentum
            #lyr.scheduler = scheduler
            
    
    
    def get_costs_updates(input, layer, learning_rate):
        '''
        For now costs are end-to-end
        '''
        x = input
        y = layer.get_output(x)
        z = layer.inverse_layer.get_output2(y, layer.W_prime) 
       
        cost = layer.loss_fn(x, z)
                      
        # compute the gradients
        gparams = T.grad(cost, self.params)
        # generate the list of SGD updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)










































