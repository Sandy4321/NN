import theano


try:
    import scipy.special
    imported_scipy = True
except ImportError:
    imported_scipy = False


class Draw_beta(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, *inputs):
        assert imported_scipy, (
            "Scipy not imported. Scipy needed for the draw_beta Op."
        )
        a, b, x = inputs
        #assert a > 0, ("Beta shape parameter a must be positive")
        #assert b > 0, ("Beta shape parameter b must be positive")
        a = theano.tensor.as_tensor_variable(a)
        b = theano.tensor.as_tensor_variable(b)
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [a,b,x], [a.type(), b.type(), x.type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        a = inputs_storage[0]
        b = inputs_storage[1]
        x = inputs_storage[2]
        z = output_storage[0]
        z[0] = scipy.special.betaincinv(a,b,x)

