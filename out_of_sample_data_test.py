from nnfs.datasets import spiral_data
from Backpropagation.Backpropagation_Loss import LossCategoricalCrossentropy
from Backpropagation.Backpropagation_Layers import DenseLayer, OutputLayerWithSoftmax
from Optimizers.Optmizers import OptmizerAdam, OptmizerSGD, OptmizerAdaGrad, OptmizerRMSProp


inputs, target_values = spiral_data(samples=1000, classes=3)

#Create Dense Layer with 2 input features and 64 output values (neurons)
hidden_layer = DenseLayer(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)


#Create a output layer with 64 input features and 3 output values (neurons)
output_layer = OutputLayerWithSoftmax(512, 3)

#Create Loss Object
loss_obj = LossCategoricalCrossentropy()

#Create optimizer
optmizer = OptmizerAdam(learning_rate=1e-3, decay=1e-4)
# optmizer = OptmizerSGD(learning_rate=1, decay=.1)

for epoch in range(10_001):
    ###NN Foward###
    hidden_layer.foward(inputs)
    output_layer.foward(hidden_layer.output)
    data_loss = loss_obj.calculate_loss(output_layer.output, target_values)
    loss_penalty = loss_obj.calculate_regularization_penalty(hidden_layer)
    output_layer_penalty = loss_obj.calculate_regularization_penalty(output_layer)
    loss = data_loss + loss_penalty + output_layer_penalty
    
    #Print progress every 100 iterations
    if not epoch % 100:
        lr = round(optmizer.current_learning_rate, 4)
        accuracy = loss_obj.calculate_accuracy(output_layer.output, target_values) 
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, data_loss: {data_loss:.3f}, loss_penalty: {loss_penalty:.3f}, loss: {loss:.3f}, learning rate: {lr}')

    ###NN Backward###
    # loss_obj.backward(output_layer.output, target_values)
    # output_layer.backward(loss_obj.derivated_inputs)
    output_layer.fast_backward(output_layer.output, target_values)
    hidden_layer.backward(output_layer.derivated_inputs)

    ###NN Optimization###
    optmizer.pre_update_params()
    optmizer.update_params(hidden_layer)
    optmizer.update_params(output_layer)
    optmizer.post_update_params()
    
print("Training complete")


#Validate the model -------------------------------

#Create out of sample dataset
oos_inputs, oos_target_values = spiral_data(samples=100, classes=3)

#Perform a foward pass of our testing data
hidden_layer.foward(oos_inputs)
output_layer.foward(hidden_layer.output)

#Calculate loss and accuracy of  predictions
loss = loss_obj.calculate_loss(output_layer.output, oos_target_values)
accuracy = loss_obj.calculate_accuracy(output_layer.output, oos_target_values)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

