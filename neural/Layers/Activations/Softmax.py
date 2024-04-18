from . import np
from .Activation import Activation

class Softmax(Activation):
    def activate(self, x):
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values)

    def activate_prime(self, error):
        n = self.outputs.shape[1]
        return np.dot((np.identity(n) - self.outputs) * self.outputs.T, error.T).T # We want final result to be in a row vecotr so take transpose
