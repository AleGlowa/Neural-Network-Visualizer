"""
Activation functions for applying to layers in NN
"""
import numpy as np


def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0.0, x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Derivatives of activation functions
def deriv_identity():
    return 1

def deriv_sigmoid(sigmoid):
    return sigmoid * (1.0 - sigmoid)

def deriv_relu(x):
    return 0.0 if x <= 0.0 else 1.0

def deriv_softmax(softmax):
    return softmax * (1.0 - softmax)

# Dictionary maps functions to their derivatives
derivatives = {'identity': deriv_identity, 'sigmoid': deriv_sigmoid, 'relu': deriv_relu, 'softmax': deriv_softmax}