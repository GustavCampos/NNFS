import numpy as np


class DenseLayerReLU:
    def __init__(self, number_of_inputs: int, number_of_neurons: int):
        #Number of inputs as first parameter to create a transposed weight matrix
        random_matrix = np.random.default_rng(1).standard_normal(size=(number_of_inputs, number_of_neurons))
        
        self.weights = 0.01 * random_matrix
        self.biases = np.zeros((1, number_of_neurons))
        
    def relu(self, output_matrix: np.array) -> np.array:
        return np.maximum(0, output_matrix)
    
    def foward_relu(self, inputs: np.array):
        #Weight matrix already tranposed in creation
        self.output = self.relu(np.dot(inputs, self.weights) + self.biases)
    
    def foward_softmax(self, inputs: np.array) -> np.array:
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        