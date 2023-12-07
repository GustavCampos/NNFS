from Backpropagation.Backpropagation_Layers import Layer


class OptmizerSGD:
    # Initialize optimizer - set settings,
# learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate:int=1.0) -> None:
        self.learning_rate = learning_rate
        
    #update parameters
    def update_params(self, layer: Layer):
        layer.weights += -self.learning_rate * layer.derivated_weights
        layer.biases += -self.learning_rate * layer.derivated_biases