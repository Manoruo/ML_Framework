from . import np
from .Activation import Activation

class TanH(Activation):
    def activate(self, x):
        return np.tanh(x)
    
    def activate_prime(self, error):
        return 1 - np.tanh(self.inputs) ** 2
