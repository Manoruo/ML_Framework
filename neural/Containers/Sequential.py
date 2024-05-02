from . import np
from ..layers.Layer import Layer
from ..layers.core.Core import Core
from ..utils.visualization import NNV
from ..metrics.accuracy import sparse_accuracy, categorical_accuracy
from tqdm import tqdm


OUTPUT_LAYER_NAME = 'output'
INPUT_LAYER_NAME = 'input'
HIDDEN_LAYER_NAME = 'hidden'


OUTPUT_LAYER_COLOR = 'red'
INPUT_LAYER_COLOR = 'darkBlue'
HIDDEN_LAYER_COLOR = 'black'

class Sequential():

    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.core_layers = [layer for layer in self.layers if isinstance(layer, Core)]
        assert len(self.core_layers) > 0, "Model must have at least one Core Layer"

        self.visualizer = None
    
    def forward(self, x):
        out = x 
        for layer in self.layers:
            out = layer.forward(out)
        return out 

    def predict(self, x):
        return self.forward(x)
    
    def fit(self, x_train, y_train, epochs, learning_rate, loss_func, accuracy='sparse'):
        # sample dimension first
        samples = len(x_train)

        # setup trackers + accuracy function
        loss_tracker = []
        accuracy_tracker = []
        accuracy_func = None 

        if accuracy == 'sparse':
            accuracy_func = sparse_accuracy
        elif accuracy == 'categorical':
            accuracy_func = categorical_accuracy

        # training loop
        for i in range(epochs):
            err = 0
            acc = 0
            for sample, y_act in zip(x_train, y_train):
                y_act, sample = np.array([y_act]), np.array([sample]) # model expects row vector, add another dimesnion to avoid errors
                
                # forward propagation (model expects a 2D list, where each element is a sample. Add another dimmension to indvidual sample to prevent it from erroring)
                y_pred = self.forward(np.array(sample))

                # compute loss (for display only)
                err += loss_func.get_loss(y_act, y_pred)
                acc = (acc + accuracy_func(y_act, y_pred)) if accuracy_func is not None else np.nan
                # backward propagation (performs gradient descent and updates weights+bias too)
                error = loss_func.get_loss_prime(y_act, y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                
            # calculate average error on all samples
            if (i + 1) % 100 == 0 or i == 0:
                # track loss and accuracy
                err /= samples
                acc /= samples
                loss_tracker.append(err)
                accuracy_tracker.append(acc)

                # display message
                print(f"Epoch {i}/{epochs}, Loss: {err:.4f}, Accuracy: {acc:.4f}")

    def display_network(self):
        if not self.visualizer:
            self.visualizer = NNV(layers_list=self._get_render_info(), spacing_nodes=5)
        self.visualizer.render()

    def _get_render_info(self):
        # this creates "descriptions" for how to draw each layer. The descriptions are in the form of a dictionary with keys ['title', 'units', 'color']

        # we need to know the first and last layer 
        first_layer = self.core_layers[0]
        last_layer = self.core_layers[-1]

        # create description info for the input layer 
        input_layer_desc =  {"title": INPUT_LAYER_NAME, "units": first_layer.get_input_size(), "color": INPUT_LAYER_COLOR}
        layersListInfo = [input_layer_desc]

        # create description for all the other layers (mention activation functions in the title for the layer)
        for x in self.layers:
            if isinstance(x, Core):
                title = HIDDEN_LAYER_NAME if x != last_layer else OUTPUT_LAYER_NAME
                color = HIDDEN_LAYER_COLOR if x != last_layer else OUTPUT_LAYER_COLOR
                units = x.get_output_size()
                layersListInfo.append({"title": title, "units": units, "color": color})
            else:
                # activation function, add on to the title for current core layer
                activation_func_name = str(type(x).__name__)
                layersListInfo[-1]['title'] += ('\n' + activation_func_name)

        return layersListInfo
      
  