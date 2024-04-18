from . import np
from .Activation import Activation

class RELU(Activation):

    def activate(self, x):
        return np.maximum(x, 0)
    
    def activate_prime(self, error):
        return np.array(self.inputs >= 0).astype('int')
