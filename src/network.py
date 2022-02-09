import pathlib
import numpy as np

from src.config_options import WeightRegularization
from src.generator import Image
from src.layer import Layer, SoftmaxLayer
from src.loss_functions import LossFunction

class Network:
    log_file = pathlib.Path("network.log")
    def __init__(self,
                 input_size : int,
                 loss_function : LossFunction,
                 weight_regularization : WeightRegularization,
                 weight_regularization_rate : float,
                 layers : list[Layer],
                 softmax : bool = False,
                 *,
                 debug: bool = False,
                 verbose: bool = False,
                 ):
        self.input_size = input_size
        self.loss_function = loss_function
        
        self.layers = layers
        
        if softmax:
            self.softmax = SoftmaxLayer()
        else:
            self.softmax = None
        
        self.debug = debug
        if self.debug:
            with open(self.log_file, "w") as file:
                # Empty the file
                pass
        self.verbose = verbose
        

    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        self._debug("##### FORWARD_PASS #####")
        self._debug(f"Input value {input_values.dumps()}")
        if input_values.size != self.input_size:
            raise ValueError(
                "input_values must have same size as self.input_size! "
                f"input_values = {input_values}, self.input_size = {self.input_size}"    
            )
        current_value = input_values
        for i, layer in enumerate(self.layers):
            current_value = layer.forward_pass(current_value)
            self._debug(f"Layer {i} - value {current_value.dumps()}")
        
        if self.softmax is not None:
            current_value = self.softmax.forward_pass(current_value)
            self._debug(f"Softmax - {current_value}")
        
        return current_value

    def backward_pass(self, predictions: np.ndarray, targets: np.ndarray):
        self._debug("##### BACKWARD_PASS #####")
        self._debug(f"Predictions: {predictions.dumps()}")
        self._debug(f"Targets: {targets.dumps()}")

        jacobi = self.loss_function.derivative(predictions, targets)
        self._debug(f"Loss jacobi matrix: {jacobi.dumps()}")

        if self.softmax is not None:
            jacobi = self.softmax.backward_pass(jacobi)
            self._debug(f"Softmax jacobi: {jacobi.dumps()}")
        
        for i, layer in enumerate(reversed(self.layers)):
            jacobi = layer.backward_pass(jacobi)
            self._debug(f"Layer {i} - jacobi {jacobi.dumps()}")
        
        for layer in self.layers:
            layer.update_weights_and_biases()
    
    def train(self, dataset: list[Image], epochs: int = 1):
        for epoch in range(epochs):
            self._debug(f"##### EPOCH {epoch} #####")
            self._verbose(f"##### EPOCH {epoch} #####")

            for i, image in enumerate(dataset):
                self._verbose(f"Input:\n{image.data.dumps()}")

                prediction = self.forward_pass(image.data)
                target = image.image_class.one_hot()

                self._debug(f"Image {i} - loss {self.loss_function.apply(prediction, target)}")
                self._verbose(f"### IMAGE {i} ###")
                self._verbose(f"Prediction: {prediction.dumps()}")
                self._verbose(f"Target:     {target.dumps()}")
                self._verbose(f"Loss:       {self.loss_function.apply(prediction, target)}")

                self.backward_pass(prediction, target)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward_pass(inputs)
    

    def _debug(self, message: str):
        if self.debug:
            with open(self.log_file, "a") as file:
                file.write(message)
    
    def _verbose(self, message: str):
        if self.verbose:
            print(message)
