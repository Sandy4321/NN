'''
Data_handling deals with all the IO issues, nonspecific to the NN model
'''

import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Data_handling:
    
    def __init__(self):
        pass
    
    def load_data(self,dataset):
        '''Load the dataset, which must be in pickled form. Will want to add a
        database retrieval function later
        
        Disclaimer: Copied straight from Montreal deep learning tutorials
        
        :type dataset: string
        :param dataset: path to dataset directory
        '''
        
        print('Loading data')
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')
    
        self.test_set_x, self.test_set_y = shared_dataset(test_set)
        self.valid_set_x, self.valid_set_y = shared_dataset(valid_set)
        self.train_set_x, self.train_set_y = shared_dataset(train_set)
    
    


    def get_corrupt(self, corruption_level):
        """ We use binary erasure noise """
        print('Corrupting test set')
        
        # Set up random number generators on CPU and GPU
        np_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        
        # Symbolic input
        input = T.dmatrix(name='input')
        
        # Define function
        corrupt = theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input
        
        # Construct expression graph
        fn = theano.function([input], corrupt)
        
        # Run function
        self.corrupt_set_x = theano.shared(np.asarray(fn(self.test_set_x.get_value()),
                                                      dtype=theano.config.floatX),
                                           borrow=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        