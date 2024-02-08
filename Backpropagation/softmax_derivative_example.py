import numpy as np
import nnfs
from Backpropagation_Layers import OutputLayerWithSoftmax as Ols
from Backpropagation_Loss import LossCategoricalCrossentropy as Lcc


nnfs.init()
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

output_layer = Ols(1, 1)
output_layer.output = softmax_outputs
output_layer.fast_backward(softmax_outputs, class_targets)
combinated_gradient = output_layer.derivated_inputs

loss_obj = Lcc()
loss_obj.backward(softmax_outputs, class_targets)
output_layer.backward(loss_obj.derivated_inputs)
separated_gradient = output_layer.derivated_inputs

print("Combinated gradient")
print(combinated_gradient)

print("Separated gradient")
print(separated_gradient)