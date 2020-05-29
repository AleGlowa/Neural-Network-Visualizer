"""
Dense Neural Network implementation in class

NN -- Neural Network
"""
from typing import Callable

import numpy as np

from value_settings import T_PRECISION
from activations import sigmoid, deriv_sigmoid
from utils import derivative
import init


class NN_Dense():
    def __init__(self, num_neurons: list, weights_init=init.with_value(), bias_init=init.with_value()) -> None:
        """Initialize NN
        
        num_neurons -- list's length represent number of layers and values within represent
                       number of neurons in each layer(including input layer, so number of features).
                       Minimum length of the list must be >= 2(1 for input and 1 for output).
        weights_init -- list containing 2d arrays or init function. Optional argument(default init with zero like).
        bias_init -- ndarray containing biases or init function for each hidden layer(default init with zero like).
        """
        assert len(num_neurons) >= 2 and all([type(neuron) == int for neuron in num_neurons]), 'Neural network must have at least 2 layers!'
        assert (callable(weights_init) or type(weights_init) == np.ndarray) and (callable(bias_init) or type(bias_init) == np.ndarray), 'weights_init and bias_init must be init functions or ndarrays!'
        weights = []
        
        if callable(weights_init):
            for m, n in zip(num_neurons[1:], num_neurons[:-1]):
                weights.append(weights_init((m, n)))
        else:
            weights = weights_init

        if callable(bias_init):
            bias = bias_init((len(weights), 1))
        else:
            assert bias_init.shape == (len(weights), 1), 'bias ndarray\'s dimensions dont\'t match with weights!'
            bias = bias_init

        self.weights, self.bias = weights, bias
        self.num_Hlayers = len(num_neurons) - 1

    def forward(self, inputs: np.array, activation: Callable[[np.array], np.array]) -> np.array:
        """Forward propagation of NN

        inputs -- input data as a 2d numpy array. First dimension is number of examples.
                  Second dimension is number of features.
        activation -- apply activation function to each hidden layer.
        """
        self.activation = activation
        outputs = inputs
        # save required values for computing gradients of weights in backward() method
        self._data_for_Wgrads = [inputs]
        for idx, (weight, bias) in enumerate(zip(self.weights, self.bias), 1):
            net_input = np.dot(weight, outputs) + bias
            outputs = activation(net_input)
            if idx != len(self.weights):
                self._data_for_Wgrads.append(outputs)

        return outputs

    def zero_grad(self):
        """Initialize weight's grads and bias's grads with zeros"""
        self.Wgrads, self.bgrads = [], []
        for weight, bias in zip(self.weights, self.bias):
            self.Wgrads.append(np.zeros_like(weight))
            self.bgrads.append(np.zeros_like(bias))

    def backward(self, loss):
        errors = [loss]
        for weight, output in zip(self.weights[-1:0:-1], self._data_for_Wgrads[-1:0:-1]):
            # CHANGE LATER deriv_sigmoid BECAUSE IS ONLY FOR SIGMOID ACTIVATION FUNC
            error = np.dot(np.transpose(weight), errors[-1]) * deriv_sigmoid(output)
            errors.append(error)

        # Initialize weight's grads and bias's grads with zeros
        self.zero_grad()

        batch_size = self._data_for_Wgrads[0].shape[1]
        for idx, (error, output, bias) in enumerate(zip(errors, self._data_for_Wgrads[-1::-1], self.bias[-1::-1]), 1):
            for ex_output, ex_error in zip(output.T, error.T):
                self.Wgrads[-idx] += np.outer(ex_error, ex_output)
            self.bgrads[-idx] = (self.bgrads[-idx] + error * bias) / batch_size
            self.Wgrads[-idx] /= batch_size

    def step(self, optim_func):
        self.weights, self.bias = optim_func((self.weights, self.bias), (self.Wgrads, self.bgrads))