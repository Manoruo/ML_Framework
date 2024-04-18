from abc import abstractmethod
from ..Layer import Layer

class Activation(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None 

    @abstractmethod
    def activate(self, x):
        pass 
    @abstractmethod
    def activate_prime(self, x):
        pass
    
    def forward(self, x):
        self.inputs = x
        self.outputs = self.activate(x)
        return self.outputs

    # no need for learning rate since we dont update any parameters here
    def backward(self, error, learning_rate):
        return error * self.activate_prime(error)
    