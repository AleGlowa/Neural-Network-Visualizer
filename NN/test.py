from NN_Dense import NN_Dense
from activations import sigmoid
import init

import numpy as np


dense_nn = NN_Dense([2, 3, 4, 1], init.normal(1.0, 2.0), init.with_value(0.05))
output = dense_nn.forward(np.array([[1, 2], [2, 3]]), sigmoid)
print(output)