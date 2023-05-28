import keras
import keras.applications
import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, array_to_img

from AbstractModel import Model


class Transfer(Model):
    def __init__(self, input_size=784, output_size=10):
        self.model = None
        self.epochs = None
        self.batch_size = None
        self.input_size = input_size
        self.output_size = output_size
        self.image_dimension = int(np.sqrt(self.input_size))

        self.model = keras.Sequential()
        self.fitted = False

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
        K.set_value(self.model.optimizer.learning_rate, 0.0001)

    # predict the output for a given input
    def predict(self, input_data):
        input_data = self.prepare_x(np.array([input_data]))
        return self.model(input_data, training=False)[0]
        # return self.model.predict(input_data, verbose=0, use_multiprocessing=True)

    # train the network

    def fit(self, x_train, y_train, epochs):

        x_train = self.prepare_x(x_train)
        y_train = self.prepare_y(y_train)

        self.epochs = epochs
        self.batch_size = 128

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
        if not self.fitted:
            self.fitted = True
            return self.fit_initial(train_generator)
        else:
            return self.fine_tune(train_generator)

    def fit_initial(self, train_generator):
        slow_down_learning_rate = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=1)
        end_training_early = EarlyStopping(monitor="accuracy", baseline=0.995, patience=3)

        return self.model.fit(train_generator.x, train_generator.y, epochs=self.epochs, batch_size=self.batch_size,
                              callbacks=[slow_down_learning_rate, end_training_early], validation_split=0.2)

    def fine_tune(self, train_generator):
        self.trans.trainable = True
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        K.set_value(self.model.optimizer.learning_rate, 0.0001)

        return self.model.fit(train_generator.x, train_generator.y, epochs=self.epochs, batch_size=self.batch_size,
                              validation_split=0.2)

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
        x_train = [np.resize(x_train[i], (28, 28, 1)) for i in range(len(x_train))]

        x = [array_to_img(x_train[i], scale=False) for i in range(len(x_train))]
        x = [x[i].resize((75, 75)) for i in range(len(x))]
        x = [x[i].convert(mode='RGB') for i in range(len(x))]
        x = [img_to_array(x[i]) for i in range(len(x))]
        return np.array(np.array(x).astype('float32') / 255)

    def init_layers_64(self):
        raise NotImplementedError()

    def init_layers_784(self):
        self.trans = keras.applications.MobileNet(
            include_top=False,
            weights='imagenet',
            input_shape=(75, 75, 3)
        )
        self.trans.trainable = False
        self.model.add(self.trans)
        self.model.add(Flatten())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

    def init_layers_proportionally(self):
        raise NotImplementedError()
