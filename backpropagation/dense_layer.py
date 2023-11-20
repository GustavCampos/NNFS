import numpy as np

class DenseLayer:
    def __init__(self, inputs_number:int, neurons_number:int) -> None:
        #Create simple weights and biases
        self.weights = np.ones(shape=(inputs_number, neurons_number))
        self.biases = np.zeros(shape=(1, neurons_number))
        
    def relu(self, output_matrix: np.array) -> np.array:
        #ReLU activation function
        return np.maximum(0, output_matrix)
    
    def foward(self, inputs:np.array) -> np.array:
        #Save inputs for backpropagation
        self.last_inputs = inputs
        
        #Base neuron result
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output
    
    def foward_relu(self, inputs:np.array) -> None:
        #Save foward result to ReLU backpropagation
        self.last_inputs_relu = self.foward(inputs)
        
        #ReLU activation
        self.output = self.relu(self.output)
        
        return self.output
    
    def backward(self, derivated_values):
        #Gradient on parameters
        self.derivated_weights = np.dot(self.last_inputs.T, derivated_values)
        self.derivated_biases = np.sum(derivated_values, axis=0, keepdims=True)
        
        #Gradient on values
        self.derivated_inputs = np.dot(derivated_values, self.weights.T)
        
    def backward_relu(self, derivated_values):
        self.backward(derivated_values)
        
        self.derivated_inputs = derivated_values.copy()
        # Zero gradient where input values were negative
        self.derivated_inputs[self.last_inputs_relu <= 0] = 0
        