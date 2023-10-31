from Dense_Layer_ReLU import DenseLayerReLU
from Loss import LossCategoricalCrossentropy
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#Create dataset
nnfs.init()
dataset_inputs, dataset_results = spiral_data(samples=100, classes=3)

#Create a dense layer w/ 2 input features and 3 output values
dense_layer = DenseLayerReLU(2, 3)

#Create a output layer w/ 3 input features and 3 output values
output_layer = DenseLayerReLU(3, 3)

#Create Loss Handler
loss_calculator = LossCategoricalCrossentropy()

#Run dataset through the layers
dense_layer.foward_relu(dataset_inputs)
output_layer.foward_softmax(dense_layer.output)

#Run output through loss function
loss = loss_calculator.calculate(output_layer.output, dataset_results)
accuracy = loss_calculator.calculate_accuracy(output_layer.output, dataset_results)

print("NN output: ")
print(output_layer.output[:5])
print("\n")
print("Loss:")
print(loss)
print("Acuracy:")
print(accuracy)
