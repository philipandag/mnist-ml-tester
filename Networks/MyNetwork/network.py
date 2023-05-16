import numpy as np

from MLP.fully_connected_layer import FullyConnectedLayer
from MLP.activation_layer import ActivationLayer
from MLP.activation_functions import tanh, tanh_derivative
from MLP.error_functions import mse, mse_derivative

from AbstractModel import Model

def target_to_vector(target):
    vector = np.zeros(10)
    vector[target] = 1.0
    vector.reshape(10, 1)
    return vector

class Network(Model):
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        self.epochs = None
        self.learning_rate = None
    
    #add a layer
    def add(self, layer):
        self.layers.append(layer)
    
    #set the loss function
    def use(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    #predict the output for a given input
    def predict(self, input_data):
        result = []

        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        result.append(output)
        
        return np.argmax(result)

    #train the network
    def fit(self, x_train, y_train):

        samples = len(x_train)

        #y_train = list(map(lambda x: target_to_vector(int(x)), y_train))

        one_percent = int(self.epochs/100)

        for i in range(self.epochs):
            err = 0
            for j in range(samples):
                output = x_train[j].reshape(1, x_train.shape[1])
                #print("starting output: ", output.shape)
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    #print(output.shape)
            
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_derivative(y_train[j], output)
                #print("starting error: ", error.shape)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)
                    #print(error.shape)
                #print("sample number: ", j)

            print("Epoch: " + str(i) + " Error: " + str(err/samples)) # average value of loss function over all samples
            
            #average error
            err /= samples
            #print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    # return the mean accuracy on the given test data and labels
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        correct_sum = 0
        for i in range(len(x_test)):
            out = self.predict(x_test[i])
            correct_sum += np.argmax(out) == np.argmax(y_test[i])
        #print("Incorrect responses: ", error_sum, "out of", len(x_test))
        return correct_sum/len(x_test)