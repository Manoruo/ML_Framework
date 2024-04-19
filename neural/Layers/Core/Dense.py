from .. import np
from .Core import Core

class Dense(Core):

    def __init__(self, num_features, num_neurons):

        # set the neurons + input features
        self.num_neurons = num_neurons
        self.num_features = num_features
        
        # weights + bias for the layer will be set later 
        self.weights = None 
        self.bias = None 

        # track what comes in and goes out of layer
        self.inputs = None 
        self.output = None

        # initalize weights + bias
        self._init_parameters(num_features, num_neurons) 

    def _init_parameters(self, num_features, num_neurons):

        #setup features + neurons
        self.num_neurons = num_neurons
        self.num_features = num_features

        # set weights and bias arrays and normalize them for faster convergence 
        self.weights = np.random.randn(num_neurons, num_features) / np.sqrt(num_features + num_neurons)
        self.bias = np.random.randn(1, num_neurons) / np.sqrt(num_features + num_neurons)


    def forward(self, x):
        # inputs @ weights.T so each row is output of neurons at a given layer + bias
        # inputs @ weights.T gets us the "weighted inputs"
        out = (x @ self.weights.T) + self.bias

        self.inputs = x 
        self.output = out 

        return out
    
    def backward(self, error, learning_rate):
        # assume the error is a row vector where each column (i) represents the "error" produced from the corresponding neuron (i) in the output layer

        dw = error.T @ self.inputs # 2D matrix of size weight.size representing error produced by each weight
        db = error # row vector representing error produced by bias
        dx = error @ self.weights # row vector representing sum of error produced by previous nueron i -> dL/dxi where i is the index of previous nueron (no need to transpose weights, since matrix multiply will use the column of the weight which is the weights associated with input from nueron i)
        
        self.weights = self.weights - (learning_rate * dw)
        self.bias = self.bias - (learning_rate * db)

        return dx 
    
    def get_input_size(self):
        return (self.num_features)
    
    def get_output_size(self):
        return (self.num_neurons)
    
    def get_num_parameters(self):
        return (self.num_features * self.num_neurons) + self.num_neurons
        