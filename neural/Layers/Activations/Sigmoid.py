from . import np
from .Activation import Activation

class Sigmoid(Activation):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, error, input_idx):
        target = self.inputs[input_idx].reshape(1, -1)
        return error * np.exp(-target) / (1 + np.exp(-target))**2
    