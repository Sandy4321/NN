'''Plot results of dropout mixture model'''
from __future__ import print_function

import fnmatch
import os

import cPickle
import matplotlib.cm as cm
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation

file_name = '/home/daniel/Code/NN/net3/train_var/sparsenodrop.pkl'

stream = open(file_name, 'r')
state = cPickle.load(stream)
stream.close()

monitor = state['monitor']
XXT = monitor['XXT']
'''
files = []

fig, ax = plt.subplots(figsize=(10,10))
for i in range(len(XXT)): 
    plt.cla()
    plt.imshow(XXT[i], interpolation='nearest', cmap='gray')
    fname = '_tmp%03d.png'%i
    print('Saving frame', fname)
    plt.savefig(fname)
    files.append(fname)

print('Making movie animation.mpg - this make take a while')
os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
#os.system("convert _tmp*.png animation.mng")

# cleanup
for fname in files: os.remove(fname)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(1)

i = 0

im = plt.imshow(XXT[i], cmap='gray')

def updatefig(*args):
    global i
    i = np.minimum(i+1,499)
    im.set_array(XXT[i])
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
plt.show()
