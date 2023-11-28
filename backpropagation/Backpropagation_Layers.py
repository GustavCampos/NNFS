import numpy as np


class Layer:
    def __init__(self, inputs_number:int, neurons_number:int) -> None:
        #Create simple weights and biases
        self.weights = np.ones(shape=(inputs_number, neurons_number))
        self.biases = np.zeros(shape=(1, neurons_number))
    
    def foward(self, inputs:np.array) -> np.array:
        #Save inputs for backpropagation
        self.last_inputs = inputs
        
        #Base neuron result
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output
    
    def backward(self, derivated_values: np.array):        
        #Gradient on parameters
        self.derivated_weights = np.dot(self.last_inputs.T, derivated_values)
        self.derivated_biases = np.sum(derivated_values, axis=0, keepdims=True)
        
        #Gradient on values
        self.derivated_inputs = np.dot(derivated_values, self.weights.T)  
     
class DenseLayer(Layer):
    def relu(self, output_matrix: np.array) -> np.array:
        #ReLU activation function
        return np.maximum(0, output_matrix)
    
    def foward(self, inputs:np.array) -> None:
        #Save foward result to ReLU backpropagation
        self.last_inputs_relu = super().foward(inputs)
        
        #ReLU activation
        self.output = self.relu(self.output)
        
        return self.output
    
    def backward(self, derivated_values):        
        self.derivated_inputs = derivated_values.copy()
        # Zero gradient where input values were negative
        
        self.derivated_inputs[self.last_inputs_relu <= 0] = 0
        
        super().backward(self.derivated_inputs)
        
class OutputLayerWithSoftmax(Layer):
    def foward(self, inputs: np.array) -> np.array:
        #Save inputs for backpropagation
        self.last_inputs = super().foward(inputs)
        
        #Get unnormalized probabilities
        exp_values = np.exp(self.last_inputs - np.max(self.last_inputs, axis=1, keepdims=True))
        
        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        #Pass to neurons backward
        self.output = probabilities
        
    def backward(self, derivated_values: np.array):
        #Create uninitialized array
        self.derivated_inputs = np.empty_like(derivated_values)
        
        #Enumerate ouputs and gradients
        for index, (output_value, derivated_value) in enumerate(zip(self.output, derivated_values)):
            #Flatten output array
            output_value = output_value.reshape(-1, 1)
            
            #Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(output_value) - np.dot(output_value, output_value.T)
            
            #Calculate sample-wise gradient and add it to the array of sample gradients
            self.derivated_inputs[index] = np.dot(jacobian_matrix, derivated_value) 
            
        #Pass to neurons backward
        super().backward(self.derivated_inputs)
                   
    #Use the derivative of loss and softmax functions together to speed up the computation
    def fast_backward(self, derivated_values: np.array, target_values: np.array):
        number_of_samples = len(derivated_values)
        
        #If target values are in one-hot vector,
        #turn into categorical labels
        if len(target_values.shape) == 2:
            target_values = np.argmax(target_values, axis=1)
            
        #Copy derivated_values shape
        self.derivated_inputs = derivated_values.copy()
        
        #Calculate gradient
        self.derivated_inputs[range(number_of_samples), target_values] -= 1

        #Normalize gradient
        self.derivated_inputs = self.derivated_inputs / number_of_samples
        
        #Pass to neurons backward
        super().backward(self.derivated_inputs)