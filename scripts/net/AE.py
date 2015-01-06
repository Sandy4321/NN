'''
An autoencoder script for the net framework
'''

from layer import Layer
from layer_train import Layer_train

# Define a layer_train object
lt = Layer_train()

# Define the default autoencoder
AE = Layer(
        v_n=784,
        h_n=500,
        layer_type='AE',
        nonlinearity='sigmoid',
        h_reg='xent',
        W_reg='L2',
        W=None,
        b=None,
        b2=None,
        mask=None)

# Now run the layer_train object on our layer
lt.get_cost_update(AE, 0.1, 'AE_SE')