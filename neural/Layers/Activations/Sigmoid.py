from . import np
from .Activation import Activation

class Sigmoid(Activation):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def activate_prime(self, error):
        return np.exp(-self.inputs) / (1 + np.exp(-self.inputs))**2
    