from . import np
from .Activation import Activation

class RELU(Activation):

    def activate(self, x):
        return np.maximum(x, 0)
    
    def backward(self, error, input_idx):
        target = self.inputs[input_idx].reshape(1, -1)
        return error * np.array(target >= 0).astype('int')
