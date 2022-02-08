import numpy as np

from src.config_options import WeightRegularization
from src.generator import Image
from src.layer import Layer, SoftmaxLayer
from src.loss_functions import LossFunction

class Network:
    def __init__(self,
                 input_size : int,
                 loss_function : LossFunction,
                 weight_regularization : WeightRegularization,
                 weight_regularization_rate : float,
                 layers : list[Layer],
                 softmax : bool = False,
                 ):
        self.input_size = input_size
        self.loss_function = loss_function
        
        self.layers = layers
        
        if softmax:
            self.softmax = SoftmaxLayer()
        else:
            self.softmax = None
        

    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        if input_values.size != self.input_size:
            raise ValueError(
                "input_values must have same size as self.input_size! "
                f"input_values = {input_values}, self.input_size = {self.input_size}"    
            )
        current_value = input_values
        for layer in self.layers:
            current_value = layer.forward_pass(current_value)
        
        if self.softmax is not None:
            current_value = self.softmax.forward_pass(current_value)
        
        return current_value

    def backward_pass(self, predictions: np.ndarray, targets: np.ndarray):
        jacobi = self.loss_function.derivative(predictions, targets)

        if self.softmax is not None:
            jacobi = self.softmax.backward_pass(jacobi)
        
        for layer in reversed(self.layers):
            jacobi = layer.backward_pass(jacobi)
        
        for layer in self.layers:
            layer.update_weights_and_biases()
    
    def train(self, dataset : list[Image]):
        for image in dataset:
            prediction = self.forward_pass(image.data)
            target = image.image_class

            self.backward_pass(prediction, target)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward_pass(inputs)
