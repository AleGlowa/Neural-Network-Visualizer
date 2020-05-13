"""
Functions for initalizing weights, biases
"""
import numpy as np

from value_settings import T_PRECISION


def zero():
    return lambda size: np.zeros(size, dtype=T_PRECISION)

def normal(mean=0.0, std=1.0):
    return lambda size: np.random.normal(mean, std, size)

def with_value(value=0.01):
    return lambda size: np.full(size, value, dtype=T_PRECISION)
