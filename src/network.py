import numpy as np

from config_options import Loss, WeightRegularization
from layer import Layer, SoftmaxLayer

class Network:
    def __init__(self,
                 input_size : int,
                 loss_function : Loss,
                 weight_regularization : WeightRegularization,
                 weight_regularization_rate : float,
                 hidden_layers : list[Layer] = None,
                 softmax : bool = False,
                 ):
        self.input_size = input_size
        if hidden_layers is None:
            self.hidden_layers = []
        else:
            self.hidden_layers = hidden_layers
        
        raise NotImplementedError


    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        if input_values.size != self.input_size:
            raise ValueError(
                "input_values must have same size as self.input_size! "
                f"input_values = {input_values}, self.input_size = {self.input_size}"    
            )
        current_value = input_values
        for layer in self.hidden_layers:
            current_value = layer.forward_pass(current_value)
        
        if self.softmax:
            raise NotImplementedError("Softmax not implemented yet, sorry")
        
        return current_value

    def backward_pass(self):
        raise NotImplementedError