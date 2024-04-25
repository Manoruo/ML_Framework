from . import np

class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        # currently this assumes you're dealing with 3D input and flattens it to 2D where the rows represnt the flatten version of an instance
        assert len(input.shape) == 3, "Flatten layer meant for 3D array (num_instance, (img_dimensions))"
        return np.reshape(input, (len(input),1, -1)).squeeze()
    

