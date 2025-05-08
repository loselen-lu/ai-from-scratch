import numpy as np
from layer import Layer
from activation import Activation

class Sequential:
    def __init__(self, layers):
        self.layers = np.array(layers)
    
    def __call__(self, inputs):
        inputs = np.array(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs