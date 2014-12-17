'''
Build an example autoencoder to be trained on the MNIST data set
'''

import numpy as np
from NN import NN
from collections import namedtuple

def main():
    
    # Instantiate an autoencoder with the given topology and nonlinearities.
    # The parameters intialisations are as per the Glorot method.
    
    
    AE = NN(
        topology=(784, 450, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot'
        )

    AE.load_data('./data/mnist.pkl.gz')
    
    AE.pretrain_params(
        method='AE',
        loss='SE',
        regulariser=('L2'),
        optimiser='SDG',
        momentum='0.1',
        scheduler='ED'
        )
    
    AE.pretrain()




























if __name__ == '__main__':
    main()







