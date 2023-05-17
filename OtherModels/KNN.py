import numpy as np
from keras.utils import to_categorical


class KNN:

    # initialize the object
    def __init__(self, input_size=784, output_size=10):
        self.X = None
        self.y = None
        self.input_size = input_size
        self.output_size = output_size
        self.K = output_size * 2

    # fit the model to the training data
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X = X_train
        self.y = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        distances = np.array(list(map(lambda x: self.distance(x, X_test), self.X)))
        sorted_idx = distances.argsort()[:self.K]  # get the indices of the K nearest neighbours (sorted by distance

        neigbour_count = {}
        for idx in sorted_idx:
            if self.y[idx] in neigbour_count:  # if class already in dict, add 1 to count
                neigbour_count[self.y[idx]] += 1
            else:  # add class to dict
                neigbour_count[self.y[idx]] = 1

        sorted_neighbours = sorted(neigbour_count.items(), key=lambda x: x[1], reverse=True)

        # convert to vector of 0s and a 1 at the index of the class with the most votes

        return to_categorical(sorted_neighbours[0][0], self.output_size)

    # return the mean accuracy on the given test data and labels
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        sum_of_correct = 0
        for i in range(len(X_test)):
            if np.argmax(self.predict(X_test[i])) == y_test[i]:
                sum_of_correct += 1
        return sum_of_correct / len(X_test)

    def distance(self, x1, x2):
        return np.sqrt(np.sum(np.power(x1 - x2, 2)))
