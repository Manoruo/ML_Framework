from . import np
from .Activation import Activation

class TanH(Activation):
    def activate(self, x):
        return np.tanh(x)
    
    def backward(self, error, input_idx):
        target = self.inputs[input_idx].reshape(1, -1)
        return error * (1 - np.tanh(target) ** 2)
