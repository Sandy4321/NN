'''
The deep framework allows us to define objects with multiple stacked
layers and to train each one greedily, followed by a fine-tuning stage.
It is also possible to skip straight to the fine-tuning stage provided
measures have ben taken to ensure proper choice of nonlinearities and
regularisation etc.
'''

from layer import Layer
from data_handling import Data_handling
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class






















