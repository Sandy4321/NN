'''
This builds on the now defunct RBM script. The basic areas can be split into
Topology, Training, Regularisation, IO, Sampling and Evaluations

@author: dew
@date: 27 Nov 14
'''

import numpy as np
import pycuda.gpuarray as ga
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.curandom import XORWOWRandomNumberGenerator as curand

class Deep:
    
    def __init__(self):
        pass

    def __init__(self, top, pcd, reg, io):
        pass
