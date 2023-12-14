from nnfs.datasets import spiral_data
from Backpropagation.Backpropagation_Loss import LossCategoricalCrossentropy
from Backpropagation.Backpropagation_Layers import DenseLayer, OutputLayerWithSoftmax
from Optimizers.Optmizers import OptmizerSGD, OptmizerAdaGrad


inputs, target_values = spiral_data(samples=100, classes=3)

#Create Dense Layer with 2 input features and 64 output values (neurons)
hidden_layer = DenseLayer(2, 64)

#Create a output layer with 64 input features and 3 output values (neurons)
output_layer = OutputLayerWithSoftmax(64, 3)

#Create Loss Object
loss_obj = LossCategoricalCrossentropy()

#Create optimizer
optmizer = OptmizerAdaGrad(decay=1e-4)

for epoch in range(10_001):
    ###NN Foward###
    hidden_layer.foward(inputs)
    output_layer.foward(hidden_layer.output)
    loss = loss_obj.calculate(output_layer.output, target_values)
    
    #Print progress every 100 iterations
    if not epoch % 100:
        lr = round(optmizer.current_learning_rate, 4)
        accuracy = loss_obj.calculate_accuracy(output_layer.output, target_values) 
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, learning rate: {lr}')

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