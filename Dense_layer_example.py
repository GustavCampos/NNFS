from Dense_Layer import DenseLayer
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()
x, y = spiral_data(samples=100, classes=3)

dense_layer1 = DenseLayer(2, 3)
dense_layer1.foward(x)

print(dense_layer1.output[:5])