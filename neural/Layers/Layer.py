from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        pass 
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, error, learning_rate):
        pass 