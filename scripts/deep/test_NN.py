# test script for NN

from NN import NN
from collections import namedtuple
import numpy as np

def test_topology_builder_pparam():
    W0 = np.ones((450,784))
    W1 = np.ones((784,450))
    b0 = np.ones((450,))
    b1 = np.ones((784,))
    b_in = np.ones((784,))
    
    params = []
    p1 = namedtuple('p1', ['W','b'])
    p1.W = W0
    p1.b = b0
    p1.b_in = b_in
    p1.nonlinearity = 'sigmoid'
    p2 = namedtuple('p2', ['W','b'])
    p2.W = W1
    p2.b = b1
    p2.nonlinearity = 'sigmoid'
    
    params.append(p1)
    params.append(p2)
 
    AE = NN(
        pparams=params,
        input_bias=True
        )

    

def test_topology_builder_defaults():
    AE = NN()


def test_topology_builder_some_data(): 
    AE = NN(
        topology=(784, 450, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot'
        )
    
    
def test_load_data(): 
    AE = NN(
        topology=(784, 450, 784),
        nonlinearities=('sigmoid','sigmoid'),
        initialisation='glorot'
        )

    AE.load_data('./data/mnist.pkl.gz')





