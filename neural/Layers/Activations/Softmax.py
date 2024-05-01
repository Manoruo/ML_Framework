from . import np
from .Activation import Activation

class Softmax(Activation):
    def activate(self, x):
        exp_values = np.exp(x)
        denom = np.sum(exp_values, axis=1).reshape(-1, 1) # this needs to be a column vector to divide each row properly 
        return exp_values / denom

    def activate_prime(self, error):
        num_outputs = error.shape[1]
        identity = np.eye(num_outputs) # get the diagnol matrix 
        softmax_result = self.outputs # get whatever the output of the softmax function was (this will be a row vector and each column would be y1, y2, y3, yn)
        return softmax_result.T * (identity - softmax_result) # transpose so we can broadcase yi to the rest of the matrix (now we have the jacobian matrix)

    def backward(self, error, learning_rate):
        # the softmax backward function produces a 2D Jacobian matrix. To find the impact of each output neron, you need to multiply the proceeding error and sum down the column to get the influence of that neuron
        return np.sum(self.activate_prime(error) * error.T, axis=0).reshape(1, -1) # reshape to give extra dimension 