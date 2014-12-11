'''
Script to train a a neural network
'''


class Train:
    
    def train_layer(self):
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
        
        if method == "AE":
            # Here we train a layer as if it were an autoencoder. For now we
            # consider the version where the weights are tied.
          #  self.method = method
          #  self.loss = loss
          #  self.regulariser = regulariser
          #  self.optimiser = optimiser
          #  self.momentum = momentum
          #  self.scheduler = scheduler
            pass                   
