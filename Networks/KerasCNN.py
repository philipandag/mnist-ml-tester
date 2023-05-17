import keras
import numpy as np
from keras.layers import *
from AbstractModel import Model


class KerasCNN(Model):
    def __init__(self, input_size=784, output_size=10):
        self.model = None
        self.epochs = None
        self.batch_size = 1
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
            optimizer='sgd',
            loss='mse',
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
        self.epochs = 1
        self.batch_size = 0

        y_train = self.prepare_y(y_train)
        x_train = self.prepare_x(x_train)
        self.model.fit(x_train, y_train, epochs=self.epochs)

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
        X = []
        for x in x_train:
            x = x.reshape(self.image_dimension, self.image_dimension, 1)
            # print(x)
            X.append(x)
        return np.array(X)

    def init_layers_64(self):
       self.model.add(keras.layers.Conv2D(
           16,
           kernel_size=(3, 3),
           activation='relu',
           input_shape=(self.image_dimension, self.image_dimension, 1), # batch of 1, image with one chanel (Grayscale)
           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
           bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
       ))
       self.model.add(keras.layers.MaxPool2D())
       self.model.add(keras.layers.Conv2D(
           32,
           kernel_size=(3, 3),
           activation='relu',
           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
           bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
       ))
       self.model.add(keras.layers.MaxPool2D(padding='same'))
       self.model.add(keras.layers.Flatten())
       self.model.add(keras.layers.Dense(
           64,
           activation='sigmoid',
           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
           bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
       ))
       self.model.add(keras.layers.Dense(
           32,
           activation='sigmoid',
           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
           bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
       ))
       self.model.add(keras.layers.Dense(
           10,
           activation='softmax',
           kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
           bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
       ))


    def init_layers_784(self):
        self.model.add(keras.layers.Conv2D(
            24,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(self.image_dimension, self.image_dimension, 1),
        ))
        self.model.add(keras.layers.MaxPool2D())
        self.model.add(keras.layers.Conv2D(
            24,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(self.image_dimension, self.image_dimension, 1),
        ))
        self.model.add(keras.layers.MaxPool2D())
        self.model.add(keras.layers.Conv2D(
            48,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(self.image_dimension, self.image_dimension, 1),
        ))
        self.model.add(keras.layers.MaxPool2D())
        self.model.add(keras.layers.Conv2D(
            24,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(self.image_dimension, self.image_dimension, 1),
        ))
        self.model.add(keras.layers.MaxPool2D())
        self.model.add(keras.layers.Conv2D(
            48,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(self.image_dimension, self.image_dimension, 1),
        ))
        self.model.add(keras.layers.MaxPool2D())
        self.model.add(keras.layers.Conv2D(
            64,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(self.image_dimension, self.image_dimension, 1),
        ))
        self.model.add(keras.layers.MaxPool2D(padding='same'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(
            256,
            activation='relu',
        ))
        self.model.add(keras.layers.Dense(
            10,
            activation='softmax',
        ))

    def init_layers_proportionally(self):
        raise NotImplementedError()

