#import numpy as np

from value_settings import DERIV_EPS

# Numeric method for calculating derivative of the function which takes one argument(net_input)
def derivative(func, arg, eps=DERIV_EPS):
    return (func(arg + eps) - func(arg)) / eps