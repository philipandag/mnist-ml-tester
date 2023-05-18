import keras
import numpy as np
from AbstractModel import Model


class KerasCNN(Model):
    def __init__(self, input_size=784, output_size=10):
        self.model = None
        self.epochs = None
        self.batch_size = None
        self.input_size = input_size
        self.output_size = output_size
        self.image_dimension = int(np.sqrt(self.input_size))

        self.model = keras.models.Sequential()

        if self.input_size == 784:
            self.init_layers_784()
        elif self.input_size == 64:
            self.init_layers_64()
        else:
            self.init_layers_proportionally()

        # SGD - stochastic gradient descent
        # MSE - mean squared error
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None
        )

    # predict the output for a given input
    def predict(self, input_data):
        input_data = self.prepare_x(np.array([input_data]))
        return self.model(input_data, training=False)[0]
        # return self.model.predict(input_data, verbose=0, use_multiprocessing=True)

    # train the network
    def fit(self, x_train, y_train):
        self.epochs = 2
        self.batch_size = 128

        y_train = self.prepare_y(y_train)
        x_train = self.prepare_x(x_train)

        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    # return the mean accuracy on the given test data and labels
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        y_test = self.prepare_y(y_test)
        x_test = self.prepare_x(x_test)
        result = self.model.evaluate(x_test, y_test)
        return result[1]

    def prepare_y(self, y):
        y = keras.utils.to_categorical(y, self.output_size)
        return y

    def prepare_x(self, x_train):
        x_train = x_train.reshape(len(x_train), self.image_dimension, self.image_dimension, 1)
        x_train = x_train.astype("float32") / 255
        # x_train = np.expand_dims(x_train, -1)
        return x_train

    def init_layers_64(self):
        self.model.add(keras.layers.Input(shape=(self.image_dimension, self.image_dimension, 1)))
        self.model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
        self.model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(self.output_size, activation="softmax"))

    def init_layers_784(self):
        self.model.add(keras.layers.Input(shape=(self.image_dimension, self.image_dimension, 1)))
        self.model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
        self.model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(self.output_size, activation="softmax"))

    def init_layers_proportionally(self):
        raise NotImplementedError()
