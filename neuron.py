import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
    
    def __call__(self, inputs):
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights) + self.bias