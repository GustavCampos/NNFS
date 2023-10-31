import numpy as np


class DenseLayer:
    def __init__(self, number_of_inputs: int, number_of_neurons: int):
        #Number of inputs as first parameter to create a transposed weight matrix
        random_matrix = np.random.default_rng(1).standard_normal(size=(number_of_inputs, number_of_neurons))
        
        self.weights = 0.01 * random_matrix
        self.biases = np.zeros((1, number_of_neurons))
    
    def foward(self, inputs: np.array):
        #Weight matrix already tranposed in creation
        self.output = np.dot(inputs, self.weights) + self.biases