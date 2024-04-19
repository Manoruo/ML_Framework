from ..Layer import Layer
from abc import abstractmethod


class Core(Layer):
    @abstractmethod
    def get_input_size(self):
        pass

    @abstractmethod
    def get_output_size(self):
        pass
        
    @abstractmethod
    def get_num_parameters(self):
        pass