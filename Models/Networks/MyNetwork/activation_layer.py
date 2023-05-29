from Models.Networks.MyNetwork.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation(self.input, derivative=True) * output_error
