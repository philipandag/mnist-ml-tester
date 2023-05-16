import keras
import numpy as np

from AbstractModel import Model


class KerasMLP(Model):
    def __init__(self):
        initializer = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)

        self.model = keras.models.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(64,), kernel_initializer=initializer,
                               bias_initializer=initializer),
            keras.layers.Dense(16, activation='relu', kernel_initializer=initializer, bias_initializer=initializer),
            keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer)
        ])

        # SGD - stochastic gradient descent
        # MSE - mean squared error
        self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

        self.epochs = 20
        self.batch_size = 128

    # predict the output for a given input
    def predict(self, input_data):
        return self.model.predict(input_data)

    # train the network
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    # return the mean accuracy on the given test data and labels
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        result = self.model.evaluate(x_test, y_test)
        return result[1]
