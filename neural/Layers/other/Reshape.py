from ..Layer import Layer
from . import np


class Reshape(Layer):
    def __init__(self, input_shape, out_shape):
        self.input_shape = input_shape
        self.out_shape = out_shape 

    def forward(self, input):
        # reshape to whatever specified 
        num_samples = len(input)
        new_shape = tuple([num_samples] + list(self.out_shape))
        return np.reshape(input, new_shape)
    
    def backward(self, error, learning_rate):
        return error.reshape(*self.input_shape)

class Flatten(Reshape):
    def __init__(self, input_shape):
        out_shape = (np.prod(input_shape),) # this would be the flattened number of elements 
        super().__init__(input_shape, out_shape)

