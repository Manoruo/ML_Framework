from . import np
from ..layers.Layer import Layer
from ..layers.core.Core import Core
from ..layers.activations.Activation import Activation
from ..layers.other import Reshape
from ..utils.visualization import NNV
from ..metrics.accuracy import sparse_accuracy, categorical_accuracy
from tqdm import tqdm
import time 
import multiprocessing

OUTPUT_LAYER_NAME = 'output'
INPUT_LAYER_NAME = 'input'
HIDDEN_LAYER_NAME = 'hidden'
RESHAPE_LAYER_NAME = 'reshape'

OUTPUT_LAYER_COLOR = 'red'
INPUT_LAYER_COLOR = 'darkBlue'
HIDDEN_LAYER_COLOR = 'black'
RESHAPE_LAYER_COLOR = 'green'

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
    
    def fit(self, x_train, y_train, epochs, learning_rate, loss_func, accuracy='sparse', batch_size=32):
 
        # setup trackers + accuracy function
        loss_tracker = []
        accuracy_tracker = []
        
        accuracy_func = categorical_accuracy if accuracy == 'categorical' else sparse_accuracy
   

        # batch
        batches = self.batch_data(x_train, y_train, batch_size)
        num_batches = len(batches)
        
        times = []

        # training loop
        for i in range(epochs):
            err = 0
            acc = 0
            

            for batch in batches:
                x, y = batch 
                

                # accumlate gradients
                for sample, y_act in zip(x, y):
                    y_act, sample = y_act.reshape(1, -1), sample.reshape(1, -1) # model expects row vector, add another dimesnion to avoid errors

                    # forward propagation
                    y_pred = self.forward(sample)

                    # backward propagation (performs gradient descent and updates weights+bias too)
                    error = loss_func.get_loss_prime(y_act, y_pred)
                    for layer in reversed(self.layers):
                        error = layer.backward(error, learning_rate)
                start = time.time()
                

                # update gradients 
                for layer in self.core_layers:
                    layer.weights = layer.weights - (learning_rate * (layer.gradient_dw / len(x)))
                    layer.bias = layer.bias - (learning_rate * (layer.gradient_db / len(x)))
                    layer.clear_gradients()

                times.append(time.time() - start)

                # compute avg loss + acc for current batch (for display only)
                y_pred = self.forward(x)
                err += loss_func.get_loss(y, y_pred)
                acc = (acc + accuracy_func(y, y_pred))  if accuracy_func is not None else np.nan
                
            # normalize metrics
            err /= num_batches
            acc /= num_batches 

            # track loss + accuracy
            loss_tracker.append(err)
            accuracy_tracker.append(acc)

            # print message
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Epoch {i}/{epochs}, Loss: {err:.4f}, Accuracy: {acc:.4f}")
            
        return loss_tracker, accuracy_tracker, times
    
    def display_network(self):
        if not self.visualizer:
            self.visualizer = NNV(layers_list=self._get_render_info(), spacing_nodes=5)
        self.visualizer.render()

    def batch_data(self, samples, labels, batch_size):
        """
            Takes the samples and corresponding labels and creates batches.
            Each batch is stored in a tuple within the returned list

        """

        assert len(samples) == len(labels), "sampes and labels are not the same size. Ensure that the labels passed correspond to the given samples"

        # determine the number of compelte batches we can make and how many sampels we'd have left over
        num_groups = len(samples) // batch_size
        remainder = len(samples) % batch_size
        
        # create batch groups (store in a tuple)
        batches = [(samples[i * batch_size: (i * batch_size) + batch_size], 
                    labels[i * batch_size: (i * batch_size) + batch_size]) 
                    for i in range(num_groups)]
        
        # add in remainder of samples + labels
        if remainder > 0:
            batches.append((samples[-remainder:], labels[-remainder:]))

        return batches
    
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
            elif isinstance(x, Activation):
                # activation function, add on to the title for current core layer
                activation_func_name = str(type(x).__name__)
                layersListInfo[-1]['title'] += ('\n' + activation_func_name)
            elif isinstance(x, Reshape):
                title = RESHAPE_LAYER_NAME
                color = RESHAPE_LAYER_COLOR
                units = np.prod(x.out_shape)
                layersListInfo.append({"title": title, "units": units, "color": color})
            else:
                raise Exception("Did not setup visual for layer type: {}".format(x.__class__.__name__))

        return layersListInfo
      
  