"""
Activation functions for applying to layers in NN
"""
import numpy as np


def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

# Derivatives of activation functions
def deriv_sigmoid(sigmoid):
    return sigmoid * (1.0 - sigmoid)