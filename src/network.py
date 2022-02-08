import numpy as np

from config_options import WeightRegularization
from layer import Layer, SoftmaxLayer
from loss_functions import LossFunction

class Network:
    def __init__(self,
                 input_size : int,
                 loss_function : LossFunction,
                 weight_regularization : WeightRegularization,
                 weight_regularization_rate : float,
                 output_layer : Layer,
                 hidden_layers : list[Layer] = None,
                 softmax : bool = False,
                 ):
        self.input_size = input_size
        if hidden_layers is None:
            self.hidden_layers = []
        else:
            self.hidden_layers = hidden_layers
        
        if softmax:
            self.softmax = SoftmaxLayer()
        else:
            self.softmax = None
        
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
        
        if self.softmax is not None:
            current_value = self.softmax.forward_pass(current_value)
        
        return current_value

    def backward_pass(self, predicted: np.ndarray, target: np.ndarray):
        error = target - predicted