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
                glorot_factor = np.sqrt(6) / np.sqrt(input_size + size)
                self.weights = self._rng.uniform(
                    -glorot_factor,
                    glorot_factor,
                    (input_size, size)
                )
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
            # Note: weights have shape = (input_size, own_size), meaning that
            # the forward pass will be performed as inputs * weights
            # This is the same shape as in the backpropagation lectures
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        self.biases = self._rng.uniform(
            bias_range[0], bias_range[1], size=size
        )

    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        weighted_inputs = np.dot(input_values, self.weights)
        biased_inputs = weighted_inputs + self.biases
        values = self.activation_function.apply(biased_inputs)

        self._cache = {"inputs": biased_inputs, "outputs": values}
        return values

    def backward_pass(self, jacobian_layer_to_loss: np.ndarray) -> np.ndarray:
        # Calculate delta Jacobian

        # Compute weight gradients for incoming weights and cache it

        # Compute bias gradients for biases, and cache it

        # Compute Jacobian from the previous layer to loss, and return it

        raise NotImplementedError
    
    def update_weights_and_biases(self):
        raise NotImplementedError


class SoftmaxLayer:
    def forward_pass(self, x : np.ndarray) -> np.ndarray:
        # Note: This could be implemented with e_x = np.exp(x) directly, but
        # this method is better for numerical stability
        # Source: https://cs231n.github.io/linear-classify/#softmax
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def backward_pass(self, x : np.ndarray) -> np.ndarray:
        raise NotImplementedError