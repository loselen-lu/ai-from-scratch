import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self, input_size, output_size):
        self.neurons = np.array([Neuron(input_size) for _ in range(output_size)])
    
    def __call__(self, inputs):
        inputs = np.array(inputs)
        return np.array([neuron(inputs) for neuron in self.neurons])