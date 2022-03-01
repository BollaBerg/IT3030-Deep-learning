import numpy as np

from src.activation_functions import ActivationFunction

class Layer:
    _cache = None
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

        self._cache = {
            "last_layer_outputs": input_values,     # = a_(l-1)
            "biased_inputs": biased_inputs,         # = z_l
            "outputs": values                       # = a_l
        }
        return values

    def backward_pass(self, jacobi_layer_to_loss: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("forward_pass must be run before backward_pass!")

        # Calculate delta Jacobian
        delta = np.multiply(
            jacobi_layer_to_loss,
            self.activation_function.derivative(self._cache.get("biased_inputs"))
        )

        # Compute weight gradients for incoming weights and cache it
        self._cache["weight_gradient"] = np.dot(delta, self._cache.get("outputs"))

        # Compute bias gradients for biases, and cache it
        self._cache["bias_gradient"] = delta

        # Compute Jacobian from the previous layer to loss, and return it
        jacobi_prev_layer = np.dot(self.weights, delta)
        return jacobi_prev_layer
    
    def update_weights_and_biases(self, regularization):
        if self._cache is None:
            raise RuntimeError("forward_pass must be run before backward_pass!")
        if self._cache.get("weight_gradient", None) is None:
            raise RuntimeError("backward_pass must be run before update_weights_and_biases!")

        self.weights += (
            -self.learning_rate
            * self._cache.get("weight_gradient")
            + regularization.apply(self.weights)
        )
        self.biases += -self.learning_rate * self._cache.get("bias_gradient")



class SoftmaxLayer:
    _cache = None
    def forward_pass(self, x : np.ndarray) -> np.ndarray:
        # Note: This could be implemented with e_x = np.exp(x) directly, but
        # this method is better for numerical stability
        # Source: https://cs231n.github.io/linear-classify/#softmax
        e_x = np.exp(x - np.max(x))
        epsilon = np.finfo(float).eps
        output = e_x / (e_x.sum(axis=0) + epsilon)

        self._cache = output
        return output
    
    def backward_pass(self, jacobi_loss: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("forward_pass must be run before backward_pass!")
        
        # The following vectorization is gotten from
        # https://stackoverflow.com/a/40576872
        output = jacobi_loss
        reshaped_output = output.reshape((-1, 1))
        jacobi = np.diagflat(output) - np.dot(reshaped_output, reshaped_output.T)

        partial_loss_softmax_inputs = np.sum(jacobi, axis=0)
        return partial_loss_softmax_inputs