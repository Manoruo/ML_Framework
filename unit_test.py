from neural.containers import Sequential
from neural.losses import CE
from neural.utils.preprocessing import OneHotEncoder
from neural.layers import RELU, Softmax, Dense, TanH, Sigmoid
import datasets.spiral_data as sd 
import copy 
import numpy as np
import sys 

def deep_copy_test():
    model = Sequential([
        RELU(),
        Dense(5, 5),
        RELU(),
        Dense(5, 5)
    ])

    # make a deep copy and change weights 
    model2 = copy.deepcopy(model)

    assert model == model2

    # change model 2 
    for i in range(len(model2.core_layers)):
        layer = model2.core_layers[i]
        
        layer.weights = np.random.random(layer.weights.shape) 
        layer.bias = np.random.random(layer.bias.shape) 

    # ensure model2 and model arent still equal 
    return model2 != model






def run_test():
    score = []

    score.append(deep_copy_test())



    for test in range(len(score)):
        print("{} Test {}".format("Passed" if score[test] else "Failed", test))
    


if __name__ == "__main__":
    run_test()