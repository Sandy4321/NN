'''
An deep autoencoder script for the deep-net framework

@author: dew
@date: 6 Jan 2013
'''

from layer import Layer
from data_handling import Data_handling
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import Image
import time

# Parameters
learning_rate = 0.1
training_size = 50000
batch_size = 50
n_train_batches = training_size/batch_size
corruption_level = 0.3
np_rng = np.random.RandomState(123)
theano_rng = RandomStreams(np_rng.randint(2 ** 30))

# Load dataset
dh = Data_handling()
dh.load_data('./data/mnist.pkl.gz')









print('Constructing expression graph')

x = T.matrix('x', dtype=theano.config.floatX)

AE = Layer(
        v_n=784,
        h_n=500,
        input=x,
        layer_type='DAE',
        nonlinearity='sigmoid',
        h_reg='xent',
        W_reg='L2',
        np_rng=np_rng,
        theano_rng=theano_rng,
        W=None,
        b=None,
        b2=None,
        mask=None)

# Do an inference sweep
#print AE.get_recon(dh.train_set_x.get_value()[30:31,:]).shape

# Now run the layer_train object on our layer
AE.load_train_params('AE_xent',
                     n_train_batches,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     pretrain_epochs=20,
                     corruption_level=corruption_level)

AE.init_weights(command='Glorot',
                nonlinearity='sigmoid')
        
### 3 Compile

index = T.lscalar()  # index to a [mini]batch

cost, updates = AE.get_cost_updates(learning_rate=AE.learning_rate)

train_layer = theano.function([index],
    cost,
    updates=updates,
    givens = {x: dh.train_set_x[index * AE.batch_size: (index + 1) * AE.batch_size]})

### 4 Train

start_time = time.clock()

print('Begining to train')
for epoch in xrange(AE.pretrain_epochs):
    # go through training set
    c = []
    for batch_index in xrange(AE.n_train_batches):
        c.append(train_layer(batch_index))
    
    end_time = time.clock()
    print('Training epoch %d, cost %5.3f, elapsed time %5.3f' % (epoch, np.mean(c), (end_time - start_time)))

image = Image.fromarray(utils.tile_raster_images(X=AE.W.get_value(borrow=True).T,
             img_shape=(28, 28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
image.save('filters_corruption_30.png')