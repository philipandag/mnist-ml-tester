
#Layer base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    # Compute the output for a given input
    def forward_propagation(self, input):
        raise NotImplementedError
    
    # Compute the input error for a given output error
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    
