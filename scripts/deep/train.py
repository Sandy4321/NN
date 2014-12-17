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
        
        layer = NN.net[layer_number]
        
        if layer.method == "AE":
            # Here we train a layer as if it were an autoencoder. In fact we train
            # layers in pairs. We don't actually summon the decoder layer, we merely
            # hallucinate it and then copy over the parameters when done training
            # the current layer.
            
            if (layer_number >= NN.num_layers/2):
                self.skip_layer = 'skip'
                print("Skipped decoder layers of AE")
            else:
                self.W_prime = layer.W.T
                self.b_prime = NN.net[NN.num_layers - layer_number - 1].b
                
                # Define encoding function
                x = T.vector('x',dtype=theano.config.floatX)
                if layer.nonlinearity == 'sigmoid':
                    encoding_function = Tnet.sigmoid(T.dot(x,layer.W) + layer.b)
                elif layer.nonlinearity == 'ultra_fast_sigmoid':
                    encoding_function = ultra_fast_sigmoid(T.dot(x,layer.W) + layer.b)
                elif layer.nonlinearity == 'hard_sigmoid':
                    encoding_function = hard_sigmoid(T.dot(x,layer.W) + layer.b)
                elif layer.nonlinearity == 'linear':
                    encoding_function = (T.dot(x,layer.W) + layer.b)
                else:
                    print("Encoding nonlinearity not supported")
    
                # Define decoding function
                h = T.vector('h',dtype=theano.config.floatX)
                if layer.nonlinearity == 'sigmoid':
                    decoding_function = Tnet.sigmoid(T.dot(h,self.W_prime) + self.b_prime)
                elif layer.nonlinearity == 'ultra_fast_sigmoid':
                    decoding_function = ultra_fast_sigmoid(T.dot(h,self.W_prime) + self.b_prime)
                elif layer.nonlinearity == 'hard_sigmoid':
                    decoding_function = hard_sigmoid(T.dot(h,self.W_prime) + self.b_prime)
                elif layer.nonlinearity == 'linear':
                    decoding_function = (T.dot(h,self.W_prime) + self.b_prime)
                else:
                    print("Encoding nonlinearity not supported")
                    
                # Define extra levels of regularisation - needs work!!!
                if layer.regulariser == 'L2':
                    regulariser = 0.0001 * (layer.W**2).sum()
                
                # Define cost
                #cost = theano.function([])
                
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
        if self.skip_layer == 'skip':
            pass
        else:
            layer = NN.net[layer_number]
        
        
            
            # Define optimisation technique
            
            
            #lyr.loss = loss
            #lyr.regulariser = regulariser
            #lyr.optimiser = optimiser
            #lyr.momentum = momentum
            #lyr.scheduler = scheduler
            
    
    
    def get_hidden_values(self, input):
        
        return theano.function([input], encoding_function)










































