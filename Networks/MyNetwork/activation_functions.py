import numpy as np


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return 1. * (x > 0)
    return x * (x > 0)


def softmax(x, derivative=False):
    if derivative:
        s = softmax(x)
        return s * (1 - s)
    return np.exp(x) / np.sum(np.exp(x))
