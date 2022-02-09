import numpy as np

class WeightRegularization:
    def __init__(self, regularization_factor):
        self.regularization_factor = regularization_factor

    def apply(self, weights: np.ndarray) -> float:
        raise NotImplementedError
    
    def derivates(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class L1(WeightRegularization):
    def apply(self, weights: np.ndarray) -> float:
        return np.sum(weights ** 2) / 2 * self.regularization_factor
    
    def derivates(self, weights: np.ndarray) -> np.ndarray:
        return weights


class L2(WeightRegularization):
    def apply(self, weights: np.ndarray) -> float:
        return np.sum(np.absolute(weights)) * self.regularization_factor
    
    def derivates(self, weights: np.ndarray) -> np.ndarray:
        return np.sign(weights)


class NoRegularization(WeightRegularization):
    def apply(self, weights: np.ndarray) -> float:
        return 0
    
    def derivates(self, weights: np.ndarray) -> np.ndarray:
        return 1