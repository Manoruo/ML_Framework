from . import np
from .Loss import Loss

class MSE(Loss):
    def get_loss(self, y_act, y_pred):
        
        if y_act.squeeze().shape != y_pred.squeeze().shape:
            raise ValueError("Arrays must have the same shape for subtraction.")

        return np.mean(np.power(y_act-y_pred, 2))
    
    def get_loss_prime(self, y_act, y_pred):
        if y_act.squeeze().shape != y_pred.squeeze().shape:
            raise ValueError("Arrays must have the same shape for subtraction.")
         
        return 2*(y_pred-y_act)/y_act.size