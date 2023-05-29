import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from AbstractModel import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Activation
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import keras.backend as K


class KerasCNNv2(Model):
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
        )
        K.set_value(self.model.optimizer.learning_rate, 0.0001)

    # predict the output for a given input
    def predict(self, input_data):
        input_data = self.prepare_x(np.array([input_data]))
        return self.model(input_data, training=False)[0]
        # return self.model.predict(input_data, verbose=0, use_multiprocessing=True)

    # train the network
    def fit(self, x_train, y_train, epochs):
        self.epochs = epochs
        self.batch_size = 128

        y_train = self.prepare_y(y_train)
        x_train = self.prepare_x(x_train)

        image_generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=[0.8, 1.0]
        )
        train_generator = image_generator.flow(
            x_train,
            y_train,
        )

        slow_down_learning_rate = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=1)
        end_training_early = EarlyStopping(monitor="accuracy", baseline=0.995, patience=10)

        return self.model.fit(train_generator.x, train_generator.y, epochs=self.epochs, batch_size=self.batch_size
                              , callbacks=[slow_down_learning_rate, end_training_early], validation_split=0.2)

    # return the mean accuracy on the given test data and labels
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        y_test = self.prepare_y(y_test)
        x_test = self.prepare_x(x_test)
        result = self.model.evaluate(x_test, y_test)
        return result[1]

    def summary(self):
        self.model.summary()

    def prepare_y(self, y):
        y = keras.utils.to_categorical(y, self.output_size)
        return y

    def prepare_x(self, x_train):
        x_train = x_train.reshape(len(x_train), self.image_dimension, self.image_dimension, 1)
        x_train = x_train.astype("float32") / 255
        # x_train = np.expand_dims(x_train, -1)
        return x_train

    def init_layers_64(self):
        raise NotImplementedError()

    def init_layers_784(self):
        self.model.add(Input(shape=(self.image_dimension, self.image_dimension, 1)))

        # 1
        self.model.add(Conv2D(
            32, kernel_size=5, kernel_regularizer=l2(0.0005)))

        # 2
        self.model.add(Conv2D(
            32, kernel_size=5, strides=1, use_bias=False))

        # 3
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(
            64, kernel_size=3, strides=1, activation="relu", kernel_regularizer=l2(0.0005)
        ))

        # 4
        self.model.add(Conv2D(
            64, kernel_size=3, strides=1, use_bias=False))

        # 5
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())

        # 6
        self.model.add(Dense(256, use_bias=False))

        # 7
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        # 8
        self.model.add(Dense(128, use_bias=False))
        # 9
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        # 10
        self.model.add(Dense(84, use_bias=False))

        # 11
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.25))

        # Out
        self.model.add(Dense(self.output_size, activation="softmax"))

    def init_layers_proportionally(self):
        raise NotImplementedError()
