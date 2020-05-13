"""
Dense Neural Network implementation in class

NN -- Neural Network
"""
from typing import Callable

import numpy as np

from value_settings import T_PRECISION
import init


class NN_Dense():
    def __init__(self, num_neurons: list, weights_init=init.with_value(), bias_init=init.with_value()) -> None:
        """Initialize NN
        
        num_neurons -- list's length represent number of layers and values within represent
                       number of neurons in each layer(including input layer, so number of features).
                       Minimum length of the list must be >= 2(1 for input and 1 for output).
        weights_init -- list containing 2d arrays or init function. Optional argument(default init with zero like).
        bias_init -- list containing biases or init function for each hidden layer(default init with zero like).
        """
        assert len(num_neurons) >= 2 and all([type(neuron) == int for neuron in num_neurons]), 'Neural network must have at least 2 layers!'
        assert (callable(weights_init) or type(weights_init) == np.ndarray) and (callable(bias_init) or type(bias_init) == np.ndarray), 'weights_init and bias_init must be init functions or ndarrays!'
        weights, bias = [], []
        
        if callable(weights_init):
            for m, n in zip(num_neurons[1:], num_neurons[:-1]):
                weights.append(weights_init((m, n)))
        else:
            weights = weights_init

        if callable(bias_init):
            bias.append(bias_init((len(weights), 1)))
        else:
            bias = bias_init

        self.weights, self.bias = weights, bias
        self.num_Hlayers = len(num_neurons) - 1

    def forward(self, inputs: np.array, activation: Callable[[np.array], np.array]) -> np.array:
        """Forward propagation of NN

        inputs -- input data as a 2d numpy array. First dimension is number of examples.
                  Second dimension is number of features.
        activation -- apply activation function to each layer.
        """
        output = inputs
        for weight, bias in zip(self.weights, self.bias):
            output = activation(np.dot(weight, output) + bias)

        return output