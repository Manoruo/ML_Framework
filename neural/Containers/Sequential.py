from . import np
from ..Layers import Layer 
from tqdm import tqdm


class Sequential():

    def __init__(self, layers=[Layer]):
        self.layers = layers
    
    def forward(self, x):
        out = x 
        for layer in self.layers:
            out = layer.forward(out)
        return out 

    def predict(self, x):
        return self.forward(x)
    
    def fit(self, x_train, y_train, epochs, learning_rate, loss_func):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in tqdm(range(epochs)):
            err = 0
            for sample, y_act in zip(x_train, y_train):

                # forward propagation (model expects a 2D list, where each element is a sample. Add another dimmension to indvidual sample to prevent it from erroring)
                y_pred = self.forward(np.array([sample]))

                # compute loss (for display only)
                err += loss_func.get_loss(y_act, y_pred)

                # backward propagation (performs gradient descent and updates weights+bias too)
                error = loss_func.get_loss_prime(y_act, y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            if (i + 1) % 100 == 0 or i == 0:
                err /= samples
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
