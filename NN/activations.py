"""
Activation functions for applying to layers in NN
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))