"""
Optimizers for learning NN
"""
import numpy as np


def gradient_descent(lr: float):
    """Updating parameters

    lr -- learning rate
    """
    def make_step(params, grads):
        """
        params -- NN's parameters to update. Two-element tuple of weights and biases.
        grads -- needed to update NN's parameters. Two-element tuple of weights' gradients and biases' gradients
        """
        upd_weights, upd_biases = [], []
        
        weights, biases = params
        Wgrads, bgrads = grads
        for weight, bias, Wgrad, bgrad in zip(weights, biases, Wgrads, bgrads):
            upd_weights.append(weight - lr * Wgrad)
            upd_biases.append(bias - lr * bgrad)

        return upd_weights, upd_biases
    
    return make_step