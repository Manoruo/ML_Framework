from .Loss import Loss
from .import np

class CE(Loss):

    def get_loss(self, y_act, y_pred):
        
           
        """
        Compute the cross-entropy loss between actual labels and predicted probabilities.

        Args:
        - y_act: Actual labels (numpy array of shape (N,))
        - y_pred: Predicted probabilities (numpy array of shape (N,))

        Returns:
        - loss: Scalar cross-entropy loss
        """

        assert y_act.shape == y_pred.shape, f"Shape of Label ({y_act.shape}) and model output ({y_pred.shape}) must match"

        # Assuming y_pred and y_act are NumPy arrays of shape (n,) or (n, m)
        y_pred = np.maximum(y_pred, 1e-10)  # Prevent any predicted values from being 0
        total_loss = np.sum(y_act * -np.log(y_pred))  # Calculate cross-entropy loss
        
        # determine the number of samples
        n = len(y_act) if len(y_act.shape) > 1 else 1  
        
        return total_loss / n 
    

    def get_loss_prime(self, y_act, y_pred):
        """
            Compute the derivative of cross-entropy loss with respect to predicted probabilities.

            Args:
            - y_act: Actual labels (numpy array of shape (N,))
            - y_pred: Predicted probabilities (numpy array of shape (N,))

            Remember equation of CE = -y_i * ln(p_i)
            taking the derivative with respect to p_i we use the chain rule and achieve the equation below. 

            Returns:
            - gradient: Gradient of cross-entropy loss with respect to predicted probabilities (numpy array of shape (N,))
        """

        pred = np.maximum(pred, 1e-10)
        return -y_act / y_pred
