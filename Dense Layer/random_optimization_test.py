from Dense_Layer_ReLU import DenseLayerReLU
from Loss import LossCategoricalCrossentropy
import numpy as np
import nnfs
from nnfs.datasets import vertical_data


#Create Dataset
nnfs.init()
dataset_inputs, dataset_results = vertical_data(samples=100, classes=3)

#Create a dense layer w/ 2 input features and 3 output values
dense_layer = DenseLayerReLU(2, 3)

#Create a output layer w/ 3 input features and 3 output values
output_layer = DenseLayerReLU(3, 3)

#Create Loss Handler
loss_calculator = LossCategoricalCrossentropy()

#Helper variables
lowest_loss = 999_999_999 #Some big initial value

best_dense_layer_weights = dense_layer.weights.copy()
best_dense_layer_biases = dense_layer.biases.copy()

best_output_layer_weights = output_layer.weights.copy()
best_dense_layer_biases = output_layer.biases.copy()

#Running the optimization
for c in range(10_000):
    #Generate new random set of weights
    seed = np.random.randint(1, 10_000)
    
    dense_layer.weights += 0.05 * np.random.default_rng(seed).standard_normal(size=dense_layer.weights.shape)
    dense_layer.biases += 0.05 * np.random.default_rng(seed).standard_normal(size=dense_layer.biases.shape)

    output_layer.weights += 0.05 * np.random.default_rng(seed).standard_normal(size=output_layer.weights.shape)
    output_layer.biases += 0.05 * np.random.default_rng(seed).standard_normal(size=output_layer.biases.shape)
    
    #Run dataset through the layers
    dense_layer.foward_relu(dataset_inputs)
    output_layer.foward_softmax(dense_layer.output)

    #Run output through loss function
    loss = loss_calculator.calculate(output_layer.output, dataset_results)
    accuracy = loss_calculator.calculate_accuracy(output_layer.output, dataset_results)
    
    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print(f"New set of weight found. Iteration: {c}, Loss: {loss}, Accuracy: {accuracy}")
        
        #Save new best weights        
        best_dense_layer_weights = dense_layer.weights.copy()
        best_dense_layer_biases = dense_layer.biases.copy()

        best_output_layer_weights = output_layer.weights.copy()
        best_output_layer_biases = output_layer.biases.copy()
        
        lowest_loss = loss
    else:
        dense_layer.weights = best_dense_layer_weights.copy()
        dense_layer.biases = best_dense_layer_biases.copy()
        
        output_layer.weights = best_output_layer_weights.copy()
        output_layer.biases = best_output_layer_biases.copy()
        
