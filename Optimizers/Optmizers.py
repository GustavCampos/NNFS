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
        
        
class OptmizerRMSProp(OptmizerSGD):
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho
        
    def update_params(self, layer: Layer):
        #If layer does not contain cache arrays,
        #Create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        #Update cache with squared current gradients
        layer.weight_cache = (self.rho * layer.weight_cache) + (1 - self.rho) * (layer.derivated_weights ** 2)
        layer.bias_cache = (self.rho * layer.bias_cache) + (1 - self.rho) * (layer.derivated_biases ** 2)
        
        layer.weights += -(self.current_learning_rate * layer.derivated_weights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases = -(self.current_learning_rate * layer.derivated_biases) / (np.sqrt(layer.bias_cache) + self.epsilon)
        
class OptmizerAdam(OptmizerRMSProp):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999) -> None:
        super().__init__(learning_rate, decay, epsilon)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    
    def update_params(self, layer: Layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        #Update momentum with current gradients
        layer.weight_momentums = (self.beta_1 * layer.weight_momentums) + (1 - self.beta_1) * layer.derivated_weights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.derivated_biases
        
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations_number + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations_number + 1))
        
        #Update cache with squared current gradients
        layer.weight_cache = (self.beta_2 * layer.weight_cache) + (1 - self.beta_2) * (layer.derivated_weights ** 2)
        layer.bias_cache = (self.beta_2 * layer.bias_cache) + (1 - self.beta_2) * (layer.derivated_biases ** 2)
        
        #Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations_number + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations_number + 1))
        
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -(self.current_learning_rate * weight_momentums_corrected) / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -(self.current_learning_rate * bias_momentums_corrected) / (np.sqrt(bias_cache_corrected) + self.epsilon)