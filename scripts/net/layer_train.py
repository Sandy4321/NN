'''
This script is designed to greedy layerwise pretrain.
'''

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tnet

from layer import Layer

class Layer_train(object):
    '''
    You should only use one layer_train object per machine. This should be
    global and shared between all layers.
    '''
    
    def __init__(self):
        pass


    ### 2 CONSTRUCT EXPRESSION GRAPH

    def get_cost_update(self, layer, learning_rate, loss_type):
            '''
            This function is based on the theano example. It computes the costs and
            a parameter update for a single training step. We consider SGD or the
            time being.
            
            :type layer: Layer object
            :param layer: current layer to optimise
            
            :type learning_rate: theano.config.floatX
            :param learning_rate: rate at which to perfrom gradient descent
            
            :type loss_type: string
            :param loss_type: loss depends on the machine
            '''
            # Define loss
            if loss_type=="AE_SE":
                pass
            elif loss_type=="AE_xent":
                # L = type something here 
            
            # need to define cost
            # need to find gradient
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            