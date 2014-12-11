'''
Build an example autoencoder to be trained on the MNIST data set
'''

import numpy as np
from net import Net
from collections import namedtuple

def main():
    
    # Instantiate an autoencoder with the given topology and nonlinearities.
    # The parameters intialisations are as per the Glorot method.
    
    '''
    W0 = np.ones((784,450))
    W1 = np.ones((450,784))
    b0 = np.ones((450,))
    b1 = np.ones((784,))
    b_in = np.ones((784,))
    
    params = []
    p1 = namedtuple('p1', ['W','b'])
    p1.W = W0
    p1.b = b0
    p1.b_in = b_in
    p2 = namedtuple('p2', ['W','b'])
    p2.W = W1
    p2.b = b1
    
    params.append(p1)
    params.append(p2)
 
    AE = Net(
        topology=(784, 450, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot',
        pparams=params,
        input_bias=True
        )
    '''    
    AE = Net(
        topology=(784, 450, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot'
        )
    
    AE.load_data('./data/mnist.pkl.gz')
    
    AE.pretrain(
        method='AE',
        loss='SE',
        regulariser=('CAE'),
        optimiser='SDG',
        momentum='0.1',
        scheduler='ED'
        )



















if __name__ == '__main__':
    main()





































