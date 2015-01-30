'''
Data_handling deals with all the IO issues, nonspecific to the NN model
'''

import gzip
import pickle
import cPickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys

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
            return shared_x, shared_y
    
        self.test_set_x,    self.test_set_y     = shared_dataset(test_set)
        self.valid_set_x,   self.valid_set_y    = shared_dataset(valid_set)
        self.train_set_x,   self.train_set_y    = shared_dataset(train_set)



    def get_corrupt(self, noise_type, corruption_level=0.5):
        """ We use binary erasure noise """
        print('Corrupting test set')
        
        # Set up random number generators on CPU and GPU
        np_rng      = np.random.RandomState(123)
        theano_rng  = RandomStreams(np_rng.randint(2 ** 30))
        
        # Symbolic input
        input       = T.dmatrix(name='input')
        
        # Define function
        # Gaussian
        if noise_type == 'gaussian':
            corrupt     = theano_rng.normal(size=input.shape, avg=0.0, std=corruption_level) + input
        # Salt and pepper
        elif noise_type == 'salt_and_pepper':
            a       = theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)
            b       = theano_rng.binomial(size=input.shape, n=1, p=0.5, dtype=theano.config.floatX)
            c       = T.eq(a,0) * b
            corrupt = (input*a) + c
        else:
            print('Invalid noise type')
            sys.exit(1)
        
        # Construct expression graph
        fn      = theano.function([input], corrupt)
        
        # Run function
        if noise_type == 'gaussian':
            self.corrupt_set_x = theano.shared(np.asarray(fn(self.test_set_x.get_value()),
                                                      dtype=theano.config.floatX),
                                           borrow=True)
        # Salt and pepper
        elif noise_type == 'salt_and_pepper':
            self.snp_set_x = theano.shared(np.asarray(fn(self.test_set_x.get_value()),
                                                  dtype=theano.config.floatX),
                                       borrow=True)
        
        else:
            print('Invalid noise type')
            sys.exit(1)

        
    
    def binarize(self, level=0.5):
        tesx     = (self.test_set_x.get_value() > level) * 1.0
        vasx    = (self.valid_set_x.get_value() > level) * 1.0
        trsx    = (self.train_set_x.get_value() > level) * 1.0
    
        self.train_set_x    = theano.shared(np.asarray(trsx, dtype=theano.config.floatX), borrow=True)
        self.valid_set_x    = theano.shared(np.asarray(vasx, dtype=theano.config.floatX), borrow=True)
        self.test_set_x     = theano.shared(np.asarray(tesx, dtype=theano.config.floatX), borrow=True)
        
        
    def shuffle_data(self):
        # Load data
        train_set_x = self.train_set_x.get_value()
        valid_set_x = self.valid_set_x.get_value()
        test_set_x  = self.test_set_x.get_value()
        big_x       = np.vstack((train_set_x, valid_set_x, test_set_x))
        
        train_set_y = self.train_set_y.get_value()[:,np.newaxis]
        valid_set_y = self.valid_set_y.get_value()[:,np.newaxis]
        test_set_y  = self.test_set_y.get_value()[:,np.newaxis]
        big_y       = np.vstack((train_set_y, valid_set_y, test_set_y))
        print big_y.shape
        
        big_data    = np.hstack((big_x, big_y))
        
        # Shuffle
        np.random.shuffle(big_data)
        train_set   = big_data[:50000,:]
        valid_set   = big_data[50000:60000,:]
        test_set    = big_data[60000:,:]   
        train_set_x, train_set_y  = np.split(train_set, [784], axis=1)
        valid_set_x, valid_set_y  = np.split(valid_set, [784], axis=1)
        test_set_x , test_set_y   = np.split(test_set, [784], axis=1)

        # Save data
        self.train_set_x    = theano.shared(np.asarray(train_set_x, dtype=theano.config.floatX), borrow=True)
        self.valid_set_x    = theano.shared(np.asarray(valid_set_x, dtype=theano.config.floatX), borrow=True)
        self.test_set_x     = theano.shared(np.asarray(test_set_x, dtype=theano.config.floatX), borrow=True)
        
        self.train_set_y    = theano.shared(np.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)
        self.valid_set_y    = theano.shared(np.asarray(valid_set_y, dtype=theano.config.floatX), borrow=True)
        self.test_set_y     = theano.shared(np.asarray(test_set_y, dtype=theano.config.floatX), borrow=True)
    
if __name__ == '__main__':
    dh = Data_handling()
    dh.load_data('./data/mnist.pkl.gz')
    dh.shuffle_data()
    dh.binarize(level=0.5)
    dh.get_corrupt('salt_and_pepper', corruption_level=0.4)
    print('Pickling data')
    stream = open('data.pkl','w')
    pickle.dump(dh, stream)
    stream.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        