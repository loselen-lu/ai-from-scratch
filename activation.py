import numpy as np

class Activation:
    @staticmethod
    def relu(inputs):
        inputs = np.array(inputs)
        return np.maximum(inputs, 0)
    
    @staticmethod
    def sigmoid(inputs):
        inputs = np.array(inputs)
        return 1 / (1 + np.exp(-inputs))