from . import np
from ..layers.Layer import Layer
from ..layers.core.Core import Core
from ..layers.activations.Activation import Activation
from ..layers.other import Reshape
from ..utils.visualization import NNV
from ..metrics.accuracy import sparse_accuracy, categorical_accuracy
from tqdm import tqdm
import copy 
import multiprocessing

OUTPUT_LAYER_NAME = 'output'
INPUT_LAYER_NAME = 'input'
HIDDEN_LAYER_NAME = 'hidden'
RESHAPE_LAYER_NAME = 'reshape'

OUTPUT_LAYER_COLOR = 'red'
INPUT_LAYER_COLOR = 'darkBlue'
HIDDEN_LAYER_COLOR = 'black'
RESHAPE_LAYER_COLOR = 'green'


NUM_WORKERS = 20


class Sequential():

    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.core_layers = [layer for layer in self.layers if isinstance(layer, Core)]
        assert len(self.core_layers) > 0, "Model must have at least one Core Layer"

        self.visualizer = None
    
    def __eq__(self, other: object):
        # check if same class
        same_class = isinstance(other, Sequential) 
        if same_class:
            # check if they have same amount of layers 
            same_num_layers = len(self.layers) == len(other.layers)
            if same_num_layers:

                # ensure they have the same layers 
                same_layers = True 
                for layer1, layer2 in zip(self.layers, other.layers):
                    if layer1 != layer2:
                        same_layers = False 
                
                return same_layers 
        return False    
        
    def forward(self, x):
        """
            Propogates input x through the layers in the 
            models layers or given array
        """
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
        accuracy_func = None 

        if accuracy == 'sparse':
            accuracy_func = sparse_accuracy
        elif accuracy == 'categorical':
            accuracy_func = categorical_accuracy
        
        # batch the data together (shuffle if specified)
        batches = self.batch_data(x_train, y_train, batch_size)
        for layer in self.core_layers: layer.set_gradient_tracking(batch_size)

        # training loop
        for i in range(epochs):
            err = 0
            acc = 0

            # loop through each batch, compute gradients and update
            for batch_idx, batch in enumerate(batches):
                print("Batch {} / {}".format(batch_idx, len(batches)))
                batch_x, y_act = batch
                num_samples = len(batch_x)

                # first do forward pass (passing through the batch)
                y_pred = self.forward(batch_x)

                err += loss_func.get_loss(y_act, y_pred)
                acc = (acc + accuracy_func(y_act, y_pred)) if accuracy_func is not None else np.nan

                # now compute loss 
                error = loss_func.get_loss_prime(y_act, y_pred)

                # compute gradients (back prop)
                self.compute_gradient(num_samples, error)
                
                # update layers 
                for layer in self.core_layers:
                    acc_dw, acc_db = layer.get_gradients()
                    dw = np.mean(acc_dw, axis=0)
                    db = np.mean(acc_db, axis=0)

                    layer.weights = layer.weights - (learning_rate * dw)
                    layer.bias = layer.bias - (learning_rate * db)
                
                # we should have gradients now 
            
            loss_tracker.append(err)
            accuracy_tracker.append(acc)
            
            # print message
            if (i + 1) % 5 == 0 or i == 0:
                print(f"Epoch {i}/{epochs}, Loss: {err / batch_size:.4f}, Accuracy: {acc / batch_size:.4f}")
            
        return loss_tracker, accuracy_tracker
    
    def compute_gradient(self, num_inputs, error):
        # error is a matrix with the error produced by the loss function for num_input samples
        # each row in error is the error produced by the loss function for a given sample 
        
        # propegate error starting from last layer working backwards 
        processes = [multiprocessing.Process(target=self.compute_gradient_helper, args=(i, error[i])) for i in range(num_inputs)]
        for process in processes: process.start()
        for process in processes: process.join()
        #print('Done', flush=True)
         

    def compute_gradient_helper(self, idx, starting_error):
        error = starting_error.reshape(1, -1)
        for layer in reversed(self.layers):
            error = layer.backward(error, idx)

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

        # create lists so we can track
        layer_grads_w = [[]for _ in range(len(self.core_layers))]
        layer_grads_b = [[] for _ in range(len(self.core_layers))]

        for sample_gradient in result:
            for i, layer_gradient in enumerate(sample_gradient):
                dw, db = layer_gradient

                if len(layer_grads_w[i]) == 0:
                    layer_grads_w[i] = dw 
                else:
                    layer_grads_w[i] += dw

                if len(layer_grads_b[i]) == 0:
                    layer_grads_b[i] = db 
                else:
                    layer_grads_b[i] += db 
        return layer_grads_w, layer_grads_b