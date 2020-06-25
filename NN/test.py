from NN_Dense import NN_Dense
from activations import sigmoid, identity
from value_settings import T_PRECISION
from cost_functions import rmse, mse, cross_entropy
from optimizers import gradient_descent
import init

import numpy as np
from math import ceil

NUM_EPOCHS = 10
BATCH_SIZE = 2

inputs = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=T_PRECISION)  # 2 examples
num_examples = inputs.shape[1]

targets = np.array([[1, 0]])  # 2 targets

num_batches = ceil(num_examples / BATCH_SIZE)

# 3 hidden layers
dense_nn = NN_Dense([inputs.shape[0], 3, 2, 1], init.with_value(0.01), init.with_value(1))

# Check predictions before training
output = dense_nn.forward(inputs, sigmoid)
print(f'Loss before training: {np.mean(cross_entropy(output, targets), 1)}')

for id_epoch in range(NUM_EPOCHS):
    for id_batch in range(num_batches):
        upper_bound = BATCH_SIZE * (id_batch + 1) if id_batch != num_batches - 1 else num_examples
        output = dense_nn.forward(inputs[:, id_batch * BATCH_SIZE : upper_bound], sigmoid)
        loss = cross_entropy(output, targets[:, id_batch * BATCH_SIZE : upper_bound])
        # calculate gradients needed for optimization algorithm
        dense_nn.backward(loss)
        dense_nn.step(gradient_descent(lr=0.02))

    # Check predictions after an epoch of training on all examples
    output = dense_nn.forward(inputs, sigmoid)
    print(f'Loss after {id_epoch+1} epoch: {np.mean(cross_entropy(output, targets), 1)}')