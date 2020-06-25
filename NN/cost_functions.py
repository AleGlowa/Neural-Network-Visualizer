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
    #sum_residuals = np.sum(sqr_residuals, axis=0, dtype=T_PRECISION)
    return np.expand_dims(np.sqrt(sqr_residuals) / 2, axis=0)


# MSE - Mean Square Error
def mse(predictions, targets):
    assert predictions.shape == targets.shape, 'Targets and predictions must be equal in length!'
    #assert predictions.shape[0] == 1, 'NN must have only one neuron in output layer!

    sqr_residuals = (predictions - targets)**2
    #sum_residuals = np.sum(sqr_residuals, axis=0, dtype=T_PRECISION)
    return np.expand_dims(sqr_residuals / 2, axis=0)

def cross_entropy(predictions, targets):
    assert predictions.shape == targets.shape, 'Targets and predictions must be equal in length!'

    # LATER CHANGE NAME OF VARIABLE!!
    logs = targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
    sum_logs = np.sum(logs, axis=0, dtype=T_PRECISION)
    return np.expand_dims(-sum_logs, axis=0)