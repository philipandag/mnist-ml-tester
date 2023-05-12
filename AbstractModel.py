import numpy as np


class Model(object):

    # initialize the object
    def __init__(self):
        raise NotImplementedError("__init__ not implemented")

    # return a list of names of hiperparameters of the model
    def get_hiperparameters(self) -> list:
        raise NotImplementedError("get_hiperparameters not implemented")

    # set the hiperparameters of the model to the values given as a dict of {"name": value}
    def set_hiperparameters(self, hiperparameters: dict):
        raise NotImplementedError("set_hiperparameters not implemented")

    # return a description of the hiperparameter with the given name (type / possible values / what will it affect / etc.)
    def get_hiperparameter_description(self, name: str) -> str:
        raise NotImplementedError("get_hiperparameter_description not implemented")

    # fit the model to the training data
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError("fit not implemented")

    # return an array of 10 floats where each float represents the probability of the corresponding digit
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict not implemented")

    # return the mean accuracy on the given test data and labels
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        raise NotImplementedError("score not implemented")
