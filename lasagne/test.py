import numpy 
from matplotlib import pyplot as plt

data = numpy.load('./models/PG0.npy')
data[:,1:] = data[:,1:]/numpy.amax(data[:,1:],axis=0)


fig = plt.figure()
plt.loglog(1-data[:,0], data[:,1:])
plt.ylim(0.1, 1.)
plt.show()
