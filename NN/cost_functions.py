"""
Functions for calculating cost/loss
"""
import numpy as np

from value_settings import T_PRECISION

# RMSE - Root Mean Square Error
def rmse(predictions, targets):
    assert predictions.shape == targets.shape, 'Targets and predictions must be equal in length!'
    #assert predictions.shape[0] == 1, 'NN must have only one neuron in output layer!'
    
    sqr_residuals = (predictions - targets)**2
    sum_residuals = np.sum(sqr_residuals, axis=1, dtype=T_PRECISION)
    return np.sqrt(sum_residuals) / targets.shape[1]


# MSE - Mean Square Error
def mse(predictions, targets):
    assert predictions.shape == targets.shape, 'Targets and predictions must be equal in length!'
    #assert predictions.shape[0] == 1, 'NN must have only one neuron in output layer!

    sqr_residuals = (predictions - targets)**2
    sum_residuals = np.sum(sqr_residuals, axis=1, dtype=T_PRECISION)
    return sum_residuals / targets.shape[1]