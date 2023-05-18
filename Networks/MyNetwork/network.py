import numpy as np
from keras.utils import to_categorical

from Networks.MyNetwork.fully_connected_layer import FullyConnectedLayer
from Networks.MyNetwork.activation_layer import ActivationLayer
from Networks.MyNetwork.activation_functions import tanh, sigmoid, relu, softmax
from Networks.MyNetwork.error_functions import mse, categorical_crossentropy

from AbstractModel import Model

class MyNetwork(Model):
    def __init__(self, input_size=64, output_size=10):
        self.layers = []
        self.loss = None
        self.epochs = None
        self.learning_rate = 0.01
        self.input_size = input_size
        self.output_size = output_size

        if self.input_size == 64:
            self.init_layers_64()
        elif self.input_size == 784:
            self.init_layers_784()
        else:
            raise Exception("Invalid input size")
    
    #add a layer
    def add(self, layer):
        self.layers.append(layer)
    
    #predict the output for a given input
    def predict(self, input_data):  # input_data = (1, 64)
        input_data = np.array(input_data).reshape(1, self.input_size)
        output = input_data.astype("float32") / 255

        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output[0]

    #train the network
    def fit(self, x_train, y_train):

        samples = len(x_train)
        y_train = self.prepare_y(y_train)

        x_train = x_train.reshape(len(x_train), 1, self.input_size)
        x_train = x_train.astype("float32") / 255

        for i in range(self.epochs):
            trained_count = 0
            true_count = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    #print(output.shape)

                trained_count += 1
                if np.argmax(output[0]) == np.argmax(y_train[j]):
                    true_count += 1

                # backward propagation
                error = self.loss(y_train[j], output, derivative=True)
                #print("starting error: ", error.shape)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)
                    #print(error.shape)


                print("\r                    \r", end="", flush=True)
                print("Epoch: ", i, "/", self.epochs, " (", j+1, "/", samples, "), acc: ", true_count/trained_count, end="")
            print("\r                    \r", end="", flush=True)
            print("Epoch: ", i, "/", self.epochs, " (", j+1, "/", samples, "), acc: ", true_count/trained_count, end="\n")


    # return the mean accuracy on the given test data and labels
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:

        x_test = x_test.reshape(len(x_test), 1, self.input_size)
        x_test = x_test.astype("float32") / 255

        correct_sum = 0
        for i in range(len(x_test)):
            output = x_test[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)

            correct_sum += np.argmax(output[0]) == y_test[i]
        #print("Incorrect responses: ", error_sum, "out of", len(x_test))
        return correct_sum/len(x_test)

    def prepare_y(self, y):
        y = to_categorical(y, self.output_size)
        return y

    def init_layers_64(self):
        self.add(FullyConnectedLayer(self.input_size, 32))
        self.add(ActivationLayer(sigmoid))
        self.add(FullyConnectedLayer(32, self.output_size))
        self.add(ActivationLayer(softmax))
        self.loss = categorical_crossentropy
        self.epochs = 10

    def init_layers_784(self):
        self.add(FullyConnectedLayer(self.input_size, 512))
        self.add(ActivationLayer(sigmoid))
        self.add(FullyConnectedLayer(512, 256))
        self.add(ActivationLayer(sigmoid))
        self.add(FullyConnectedLayer(256, 128))
        self.add(ActivationLayer(sigmoid))
        self.add(FullyConnectedLayer(128, 10))
        self.add(ActivationLayer(softmax))
        self.loss = categorical_crossentropy
        self.epochs = 2