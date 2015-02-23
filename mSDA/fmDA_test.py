'''
fmDA_test.py
'''
from fmDA import fmDA
import numpy as np
import numpy.random as rp
import time
import cPickle
import gzip
import utils
from PIL import Image
import sys


fmda = fmDA()
# Note the data is stored row-wise and the fmDA takes it column-wise
print('Loading data')
T, V, test  = fmda.load('../net/data/mnist.pkl.gz')
X           = np.vstack((T[0],V[0])).T
Xtest       = test[0].T
print('Computing layers')
#params_fmSDA= fmda.fmSDA('fmSDA',X,3)
params_mSDA = fmda.fmSDA('mSDA',X,2)
#loss_fmSDA  = fmda.test(Xtest,params_fmSDA)
loss_mSDA   = fmda.test(Xtest,params_mSDA)
loss_mSDA



'''
num_imgs = 400
index = rp.choice(W.shape[1], num_imgs, replace=False)
img = W[:,index]
image = Image.fromarray(utils.tile_raster_images(X=img.T,
         img_shape=(28,28), tile_shape=(20, 20),
         tile_spacing=(1, 1)))
image.save('hypSDA.png')
'''