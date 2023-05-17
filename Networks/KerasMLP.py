import keras
import numpy as np

from AbstractModel import Model


class KerasMLP(Model):
    def __init__(self, input_size=784, output_size=10):
        self.model = None
        self.epochs = None
        self.batch_size = None
        self.input_size = input_size
        self.output_size = output_size

        self.model = keras.models.Sequential()

        if self.input_size == 784:
            self.init_layers_784()
        elif self.input_size == 64:
            self.init_layers_64()
        else:
            self.init_layers_proportionally()

    # predict the output for a given input
    def predict(self, input_data):
        input_data = np.array(input_data).reshape(1, self.input_size)
        input_data = input_data.astype("float32") / 255
        return self.model(input_data, training=False)[0]
        # return self.model.predict(input_data, verbose=0, use_multiprocessing=True)

    # train the network
    def fit(self, x_train, y_train):
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

        self.epochs = 10
        self.batch_size = 128

        y_train = self.prepare_y(y_train)
        x_train = x_train.astype("float32") / 255

        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    # return the mean accuracy on the given test data and labels
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        x_test = x_test.astype("float32") / 255
        y_test = self.prepare_y(y_test)

        result = self.model.evaluate(x_test, y_test)
        return result[1]

    def prepare_y(self, y):
        y = keras.utils.to_categorical(y, self.output_size)
        return y

    def init_layers_64(self):
        self.model.add(keras.layers.Dense(
            64,
            activation='sigmoid',
            input_shape=(self.input_size,),
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
            16,
            activation='sigmoid',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            self.output_size,
            activation='softmax',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))

    def init_layers_784(self):
        self.model.add(keras.layers.Dense(
            512,
            activation='relu',
            input_shape=(self.input_size,),
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            256,
            activation='relu',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            128,
            activation='relu',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            self.output_size,
            activation='softmax',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))

    def init_layers_proportionally(self):
        self.model.add(keras.layers.Dense(
            512,
            activation='sigmoid',
            input_shape=(self.input_size,),
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            self.input_size * 3 // 4,
            activation='sigmoid',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            self.input_size // 2,
            activation='sigmoid',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            self.input_size // 4,
            activation='sigmoid',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
        self.model.add(keras.layers.Dense(
            self.output_size,
            activation='softmax',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            bias_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        ))
