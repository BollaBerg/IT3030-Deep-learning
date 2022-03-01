import numpy as np
from pkg_resources import AvailableDistributions

class ActivationFunction:
    def apply(self, x : float) -> float:
        raise NotImplementedError
    
    def derivative(self, x : float) -> float:
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def apply(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: float) -> float:
        applied_sigmoid = self.apply(x)
        return applied_sigmoid * (1 - applied_sigmoid)


class Tanh(ActivationFunction):
    def apply(self, x: float) -> float:
        exp = np.exp(x)
        neg_exp = np.exp(-x)
        return (exp - neg_exp) / (exp + neg_exp)
    
    def derivative(self, x: float) -> float:
        applied_tanh = self.apply(x)
        return 1 - np.power(applied_tanh, 2)


class Relu(ActivationFunction):
    def apply(self, x: float) -> float:
        return np.maximum(x, 0)
    
    def derivative(self, x: float) -> float:
        # While this could be a simple `return x > 0`, but the following
        # vectorizes the method, meaning it can be used if x is np.ndarray
        return (x > 0) * 1


class Linear(ActivationFunction):
    def apply(self, x: float) -> float:
        return x
    
    def derivative(self, x: float) -> float:
        if isinstance(x, np.ndarray):
            return np.ones(x.shape)
        else:
            return 1