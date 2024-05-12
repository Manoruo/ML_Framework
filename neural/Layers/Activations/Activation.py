from abc import abstractmethod
from ..Layer import Layer

class Activation(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None 

    @abstractmethod
    def activate(self, x):
        pass 
    
    def forward(self, x):
        self.inputs = x
        self.outputs = self.activate(x)
        return self.outputs