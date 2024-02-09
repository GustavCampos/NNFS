import numpy as np
from Backpropagation_Layers import Layer

class Loss():
    def calculate_loss(self, nn_predictions:np.array, target_values:np.array) -> float:
        sample_loses = self.foward(nn_predictions, target_values)
        data_loss = np.mean(sample_loses)

        return data_loss
    
    def clip_prediction_dataset(self, prediction_dataset):
        clip_max_value = (1 - 1e-7)
        clip_min_value = (1e-7)
        
        return np.clip(
            prediction_dataset,
            clip_min_value,
            clip_max_value
        )
    
    def calculate_accuracy(self, nn_predictions:np.array, target_values:np.array) -> float:
        predictions_categorical_label = np.argmax(nn_predictions, axis=1)
        real_values_in_onehot_vectors = (len(target_values.shape) == 2)
        
        if real_values_in_onehot_vectors:
            #Guarantee target_values always are in categorical labels
            target_values = np.argmax(target_values, axis=1)
            
        return np.mean(predictions_categorical_label == target_values)
    
    def calculate_regularization_penalty(self, layer: Layer) -> float:
        regularization_penalty = 0
        
        #L1 regularization________________________________________________________________
        #Weights
        if layer.weight_regularizer_l1 > 0:
            regularization_penalty += layer.weight_regularizer_l1 * np.sum(np.absolute(layer.weights))
            
        #Biases
        if layer.bias_regularizer_l1 > 0:
            regularization_penalty += layer.bias_regularizer_l1 * np.sum(np.absolute(layer.biases))
            
        #L2 regularization________________________________________________________________
        #Weights
        if layer.weight_regularizer_l2 > 0:
            regularization_penalty += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
 
        #Biases
        if layer.bias_regularizer_l2 > 0:
            regularization_penalty += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
            
        return regularization_penalty
    
class LossCategoricalCrossentropy(Loss):
    def foward(self, nn_prediction:np.array, target_values:np.array) -> np.array:
        #This guarantees that nn output values never reaches 0
        #to avoid log(0) Error
        clipped_nn_prediction = self.clip_prediction_dataset(nn_prediction)
        
        #There are two types of real_values notation
        #Categorical labels: 
        #   list with only correct neuron index
        #   Ex: 3 neurons = [1, 0, 2, ..., 1]
        real_values_in_categorical_labels = (len(target_values.shape) == 1)
        #One-Hot vector:
        #   list of vector with only one element '1'
        #   Ex: 3 neurons = [[0, 1, 0], [1, 0, 0], ..., [0, 0, 1]]
        real_values_in_onehot_vectors = (len(target_values.shape) == 2)
        
        if real_values_in_categorical_labels:
            correct_confidences = clipped_nn_prediction[range(len(nn_prediction)), target_values]
        elif real_values_in_onehot_vectors:
            correct_confidences = np.sum(clipped_nn_prediction * target_values, axis=1)
        else:
            raise ValueError("target_values does not have a valid shape")
 
        return -np.log(correct_confidences)

    def backward(self, derivated_values, target_values):
        number_of_samples = len(derivated_values)
        
        #Number of labels in every sample
        #Using the first line to count
        labels = len(derivated_values[0])
        
        # If labels are sparse (categorical labels), turn them into one-hot vector
        if len(target_values.shape) == 1:
            target_values = np.eye(labels)[target_values]
            
        #Calculate gradient
        self.derivated_inputs = -(target_values / derivated_values)
        
        #Normalize gradient
        self.derivated_inputs = self.derivated_inputs / number_of_samples
  