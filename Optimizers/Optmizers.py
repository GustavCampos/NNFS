import numpy as np
from Backpropagation.Backpropagation_Layers import Layer


class OptmizerSGD:
    # Initialize optimizer - set settings,
# learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate:float=1.0, decay:float=0, momentum:float=0) -> None:
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations_number = 0
        
    #update parameters
    def update_params(self, layer: Layer):
        weights_aditional = -(self.current_learning_rate * layer.derivated_weights)
        biases_aditional = -(self.current_learning_rate * layer.derivated_biases)
        
        #Simple optimization
        layer.weights += weights_aditional
        layer.biases += biases_aditional
        
        #Adding momentum to optimization
        if self.momentum:        
            #Create variables to retain momentum information
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                
                #Momentum variables are only create here
                #not weight_momentums means not bias_momentums either
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weight_retain_factor = self.momentum * layer.weight_momentums
            bias_retain_factor = self.momentum * layer.bias_momentums
            
            #Saving momentums
            layer.weight_momentums = (weights_aditional + weight_retain_factor)
            layer.bias_momentums = (biases_aditional + bias_retain_factor)
            
            #Adding momentum factor to layer
            layer.weights += weight_retain_factor
            layer.biases += bias_retain_factor
        
    #Call once before any parameter updates
    def pre_update_params(self):
        #Run only if decay is != 0
        if self.decay:
            decay_multiplier = (1 / (1 + (self.decay * self.iterations_number)))
            self.current_learning_rate = self.initial_learning_rate * decay_multiplier
            
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations_number += 1
        

class OptmizerAdaGrad(OptmizerSGD):
    def __init__(self, learning_rate: float = 1, decay: float = 0, epsilon:float=1e-7) -> None:
        #Guarantees the not use of momentum variable
        super().__init__(learning_rate, decay, 0)
        self.epsilon = epsilon
        
    def update_params(self, layer: Layer):
        #Create cache arrays
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        #Update cache with squared current gradients
        layer.weight_cache += layer.derivated_weights ** 2
        layer.bias_cache += layer.derivated_biases ** 2
        
        weights_aditional = -(self.current_learning_rate * layer.derivated_weights)
        biases_aditional = -(self.current_learning_rate * layer.derivated_biases)
        
        layer.weights += weights_aditional / (np.sqrt(layer.weight_cache) + self.epsilon) 
        layer.biases += biases_aditional / (np.sqrt(layer.bias_cache) + self.epsilon)
        
        