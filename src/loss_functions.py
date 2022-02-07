import numpy as np

class LossFunction:
    def apply(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError
    
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CrossEntropy(LossFunction):
    def apply(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        # Normalize values in case they do not sum to 1
        normalized_pred = predictions / predictions.sum()
        normalized_targets = targets / targets.sum()
        # Avoid log(0)
        epsilon = np.finfo(float).eps
        return -np.sum(normalized_targets * np.log2(normalized_pred + epsilon))
    
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return predictions - targets


class MeanSquaredError(LossFunction):
    def apply(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        if isinstance(predictions, np.ndarray):
            denominator = predictions.shape[0]
        else:
            denominator = 1
        sum_ = np.sum((predictions - targets) ** 2)
        return sum_ / denominator
    
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if isinstance(predictions, np.ndarray):
            denominator = predictions.shape[0]
        else:
            denominator = 1

        sums = np.array([
            np.sum(predictions - targets) * pred for pred in predictions
        ])
        return sums / denominator
