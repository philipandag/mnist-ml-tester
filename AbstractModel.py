import numpy as np


class Model(object):

    # initialize the object
    def __init__(self):
        raise NotImplementedError("__init__ not implemented")

    # fit the model to the training data
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError("fit not implemented")

    # return an array of 10 floats where each float represents the probability of the corresponding digit
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict not implemented")

    # return the mean accuracy on the given test data and labels
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        raise NotImplementedError("score not implemented")
