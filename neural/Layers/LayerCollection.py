from .Layer import Layer
from ..losses.Loss import Loss
from .Gradients import GradientCollection
from copy import deepcopy

class LayerCollection():

    def __init__(self, layers: list[Layer], loss_func):
        self.layers = tuple(deepcopy(layers))
        self.loss_func: Loss = loss_func
        

    def forward(self, x):
        out = x 
        for layer in self.layers:
            out = layer.forward(out)
        return out 

    def get_gradient(self, x, y):

        # forward propagation to get current prediction
        y_pred = self.forward(x)

        # backward propagation to retrieve gradients
        dx = self.loss_func.get_loss_prime(y, y_pred)
        gc = GradientCollection()
        for layer in reversed(self.layers):
            gradients = layer.backward(dx)
            dx = gradients[0]
            gc.append(gradients)

        return gc 
        