from .. import np
from .. import mp
from .Core import Core

class Dense(Core):
    # you can use * to specify how many features. I want to be able to just specify the neurons and have the model automaticlaly figure out input sizes
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
        
        #self.gradient_dw_queue = mp.Queue()
        #self.gradient_db_queue = mp.Queue()

        # initalize weights + bias
        self._init_parameters(num_features, num_neurons) 

    def _init_parameters(self, num_features, num_neurons):

        #setup features + neurons
        self.num_neurons = num_neurons
        self.num_features = num_features

        # set weights and bias arrays and normalize them for faster convergence 
        self.weights = np.random.randn(num_neurons, num_features) / np.sqrt(num_features + num_neurons)
        self.bias = np.random.randn(1, num_neurons) / np.sqrt(num_features + num_neurons)

        # set gradient tracking
        self.gradient_dw = np.zeros(self.weights.shape)
        self.gradient_db = np.zeros(self.bias.shape)



    def forward(self, x):
        # inputs @ weights.T so each row is output of neurons at a given layer + bias
        # inputs @ weights.T gets us the "weighted inputs"
        out = (x @ self.weights.T) + self.bias

        self.inputs = x 
        self.output = out 

        return out
    
    def backward(self, error):
        # assume the error is a row vector where each column (i) represents the "error" produced from the corresponding neuron (i) in the output layer

        dw = error.T @ self.inputs # 2D matrix of size weight.size representing error produced by each weight
        db = error # row vector representing error produced by bias
        dx = error @ self.weights # row vector representing sum of error produced by previous nueron i -> dL/dxi where i is the index of previous nueron (no need to transpose weights, since matrix multiply will use the column of the weight which is the weights associated with input from nueron i)
        
        #self.weights = self.weights - (learning_rate * dw)
        #self.bias = self.bias - (learning_rate * db)
        self.gradient_dw += dw 
        self.gradient_db += db 

        #self.gradient_dw_queue.put(dw)
        #self.gradient_db_queue.put(db)

        return (dx, np.nan, np.nan) 

    def clear_gradients(self):
        self.gradient_dw = np.zeros(self.weights.shape)
        self.gradient_db = np.zeros(self.bias.shape)
    
    def get_input_size(self):
        return (self.num_features)
    
    def get_output_size(self):
        return (self.num_neurons)
    
    def get_num_parameters(self):
        return (self.num_features * self.num_neurons) + self.num_neurons
        