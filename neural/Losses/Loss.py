from abc import ABC, abstractmethod

class Loss(ABC):

    @abstractmethod
    def get_loss(self, y_act, y_pred):
        pass 
    
    @abstractmethod
    def get_loss_prime(self, y_act, y_pred):
        pass
    