import numpy as np

class DenseLayerWithBackpropagation:
    def __init__(self, number_of_inputs: int, number_of_neurons: int) -> None:
        #Create neurons with weights value 1
        self.weights = np.ones((number_of_inputs, number_of_neurons))
        
        #Create Biases with value 0
        self.biases = np.zeros((1, number_of_neurons))
        
    def relu(self, output_matrix: np.array) -> np.array:
        return np.maximum(0, output_matrix)
    
    def foward_relu(self, inputs: np.array, save_inside: bool =False) -> np.array:
        #Saving inputs for backpropagation
        self.last_inputs = inputs
        
        #Get product operation result
        #Save raw output for backpropagation
        self.raw_output = np.dot(inputs, self.weights) + self.biases
        
        #Pass trought relu activation
        self.output = self.relu(self.raw_output)
        return self.output
        
    def backward(self, next_layer_derivatives: np.array) -> np.array:
        #ReLU activation's derivative with chain rule applied
        self.relu_derivative = next_layer_derivatives.copy()
        # Zero gradient where input values were negative
        self.relu_derivative[self.raw_output <= 0] = 0
        
        #Gradient on parameters
        self.weights_derivative = np.dot(self.last_inputs.T, self.relu_derivative)
        self.biases_derivative = np.sum(self.relu_derivative, axis=0, keepdims=True)

        #Gradient on values
        self.inputs_derivative = np.dot(self.relu_derivative, self.weights.T)

