import numpy as np
from nnfs.datasets import spiral_data
from Backpropagation_Layers import DenseLayer, OutputLayerWithSoftmax
from Backpropagation_Loss import LossCategoricalCrossentropy


inputs, target_values = spiral_data(samples=100, classes=3)

#Create hidden layer with 2 inputs and 3 outputs (neurons)
hidden_layer = DenseLayer(2, 3)

#Create output layer with 3 inputs and 3 outputs (neurons)
output_layer = OutputLayerWithSoftmax(3, 3)

#Create Loss function object
loss_obj = LossCategoricalCrossentropy()


### Running NN ###
hidden_layer.foward(inputs)
output_layer.foward(hidden_layer.output)
loss = loss_obj.calculate(output_layer.output, target_values)
accuracy = loss_obj.calculate_accuracy(output_layer.output, target_values)

#NN output
print(f"Loss: {loss}, Accuracy: {accuracy}")
print("NN output:")
print(output_layer.output[:5])

#Backward pass
output_layer.fast_backward(output_layer.output, target_values)
hidden_layer.backward(output_layer.derivated_inputs)

# Print gradients
print("Hidden layer gradients")
print("Weights:")
print(hidden_layer.derivated_weights)
print("Biases:")
print(hidden_layer.derivated_biases)

print("Output layer gradients")
print("Weights:")
print(output_layer.derivated_weights)
print("Biases:")
print(output_layer.derivated_biases)
