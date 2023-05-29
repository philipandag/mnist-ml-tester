import numpy as np


def mse(y_true, y_pred, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true) / len(y_true)
    return np.mean(np.power(y_true - y_pred, 2))


def categorical_crossentropy(y_true, y_pred, derivative=False):
    if derivative:
        return y_pred - y_true
    return - np.sum(y_true * np.log(y_pred))
