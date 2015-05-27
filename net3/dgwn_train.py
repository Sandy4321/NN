'''DWGN train file'''

import numpy

from DGWN import Dgwn
from train import Train

args = {
    'algorithm' : 'SGD',
    'RMScoeff' : 0.9,
    'RMSreg' : 1e-3,
    'mode' : 'training',
    'learning_type' : 'classification',
    'num_classes' : 10,
    'train_cost_type' : 'nll',
    'valid_cost_type' : 'accuracy',
    'layer_sizes' : (784, 800, 800, 10),
    'nonlinearities' : ('ReLU', 'ReLU', 'SoftMax'),
    'data_address' : './data/mnist.pkl.gz',
    'binarize': False,
    'learning_rate' : 1e-4,
    'lr_multipliers' : {'b' : 2., 'R' : 1e0},
    'learning_rate_margin' : (0,200,300),
    'learning_rate_schedule' : ((1.,),(0.5,0.1),(0.05,0.01,0.005,0.001)),
    'momentum' : 0.9,
    'momentum_ramp' : 0,
    'batch_size' : 128,
    'num_epochs' : 500,
    'prior_variance' : 1e-3,
    'num_components' : 2,
    'num_samples' : 1,
    'norm' : None,
    'max_row_norm' : None,
    'sparsity' : None, 
    'dropout_dict' : None,
    'cov' : False,
    'validation_freq' : 5,
    'save_freq' : 50,
    'save_name' : 'pkl/DGWNreg.pkl'
    }

if args['sparsity'] != None:
    # Just for now until we sort ourselves out. Promise xx
    c = 1
    N = args['layer_sizes'][0]
    Y = args['layer_sizes'][-1]
    t = total_weights(args['layer_sizes'])
    H = layer_from_sparsity(N, Y, t, 1., 1-args['sparsity'], c)
    args['layer_sizes'] = write_neurons(N, H, Y, c)
    args['connectivity'] = (1.,) + (1-args['sparsity'],)*c + (1.,)
    args['nonlinearities'] = ('ReLU',) + ('ReLU',)*c + ('SoftMax',)
    print args['layer_sizes'], args['connectivity']

if args['dropout_dict'] == True:
    dropout_dict = {}
    for i in numpy.arange(len(args['nonlinearities'])):
        name = 'layer' + str(i)
        shape = (args['layer_sizes'][i],1)
        if i == 0:
            # Need to cast to floatX or the computation gets pushed to the CPU
            prior = 0.8*numpy.ones(shape).astype(Tconf.floatX)
        else:
            prior = 0.5*numpy.ones(shape).astype(Tconf.floatX)
        sub_dict = { name : {'seed' : 234,
                             'values' : prior}}
        dropout_dict.update(sub_dict)
    args['dropout_dict'] = dropout_dict

tr = Train()
tr.build(Dgwn, args)
tr.load_data(args)
monitor = tr.train(args)
