# tryout.py

import numpy as np
import theano as T



a = [[1,2,3],[4,5,6]]
a = np.asarray(a)
b = T.shared(value=a, name='b', borrow=True)

print a
print b
print b.get_value()

print b**2

x = T.tensor.matrix('x', dtype=T.config.floatX)
y = T.tensor.scalar('y', dtype=T.config.floatX)
c = x**y
fn = T.function([x,y], c)
print fn(b,2)