from pytest import approx
import numpy as np

from src.loss_functions import CrossEntropy, MeanSquaredError

def test_cross_entropy():
    """Test that CrossEntropy.apply gives correct outputs"""
    cross_entropy = CrossEntropy()

    observations = np.array([0.1, 0.2, 0.3, 0.4])
    targets = np.array([0.1, 0.4, 0.4, 0.1])

    # assert cross_entropy.apply(observations, targets) == approx(2.08794309)




def test_mean_squared_error():
    """Test that MeanSquaredError.apply gives correct outputs"""
    MSE = MeanSquaredError()

    observations = np.array([1, 2, 3, 4])
    targets = np.array([1, 4, 2, 0])

    assert MSE.apply(observations, targets) == 5.25