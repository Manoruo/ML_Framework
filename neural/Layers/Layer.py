from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        pass 
    
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, error, learning_rate):
        pass 