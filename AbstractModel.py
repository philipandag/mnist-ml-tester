import numpy as np


class Model(object):

    # initialize the object
    def __init__(self):
        raise NotImplementedError("__init__ not implemented")

    # fit the model to the training data
    # return history of training in a dictionary [loss, accuracy, val_loss, val_accuracy]
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int) -> np.ndarray:
        raise NotImplementedError("fit not implemented")

    # return an array of n floats where each float represents the probability of the corresponding digit
    # where n is the number of classes in the training data
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict not implemented")

    # return the mean accuracy on the given test data and labels
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        raise NotImplementedError("score not implemented")

    # prints a summary of the model
    def summary(self) -> None:
        raise NotImplementedError("summary not implemented")


class DummyModel(Model):

    def __init__(self):
        self.X = None
        self.y = None
        self.fitted = False
        print("Dummy model initialized")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X = X_train
        self.y = y_train
        self.fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise Exception("Model not fitted")
        return np.array(np.random.rand(1, len(np.unique(self.y))))

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        sum_of_correct = 0
        for i in range(len(X_test)):
            if np.argmax(self.predict(X_test[i])) == y_test[i]:
                sum_of_correct += 1
        return sum_of_correct / len(X_test)

    def summary(self):
        print("Dummy model - summary")
