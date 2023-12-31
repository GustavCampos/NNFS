import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

n1_weights = [[0.2, 0.8, -0.5, 1.0],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]
n1_biases = [2.0, 3.0, 0.5]

n2_weights = [[0.1, -0.14, 0.5],
              [-0.5, 0.12, -0.33],
              [-0.44, 0.73, -0.13]]
n2_biases = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(n1_weights).T) + n1_biases
layer2_outputs = np.dot(layer1_outputs, np.array(n2_weights).T) + n2_biases

print(layer2_outputs)