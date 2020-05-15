from NN_Dense import NN_Dense
from activations import sigmoid, identity
from value_settings import T_PRECISION
from cost_functions import rmse, mse
import init

import numpy as np


inputs = np.array([[1, 2, 5, 4], [2, 3, 5, -3], [5, 3, 1, 0]], dtype=T_PRECISION)  # 4 examples
targets = np.array([[55.5, 2.4, 32.2, 52.1]])  # 4 targets

# 4 hidden layers
dense_nn = NN_Dense([inputs.shape[0], 4, 3, 2, 1], init.normal(1.0, 2.0), init.with_value(0.05))

outputs = dense_nn.forward(inputs, sigmoid)
loss = rmse(outputs, targets)
print(loss)