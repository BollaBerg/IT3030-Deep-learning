import numpy as np

from src.activation_functions import ActivationFunction

class Layer:
    def __init__(self,
                 size : int,
                 input_size : int,
                 activation_function : ActivationFunction,
                 initial_weight_range : tuple[float, float] | str,
                 learning_rate : float,
                 bias_range : tuple[float, float] = (0, 1)):
        self._rng = np.random.default_rng()
        if isinstance(initial_weight_range, str):
            if initial_weight_range.lower() == "glorot":
                raise NotImplementedError("Glorot is not yet implemented")
            else:
                raise ValueError(
                    "If initial_weight_range is a string, it must be 'glorot'!"
                    f" initial_weight_range was {initial_weight_range}"
                )
        else:
            self.weights = self._rng.uniform(
                initial_weight_range[0],
                initial_weight_range[1],
                (input_size, size)
            )
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        self.biases = self._rng.uniform(
            bias_range[0], bias_range[1], size=size
        )

    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        weighted_inputs = np.dot(input_values, self.weights)
        biased_inputs = weighted_inputs + self.biases
        values = self.activation_function.apply(biased_inputs)
        return values

    def backward_pass(self) -> np.ndarray:
        raise NotImplementedError

class SoftmaxLayer(Layer):
    pass